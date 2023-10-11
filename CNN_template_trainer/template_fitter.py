"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
# pylint: disable=W0201
import sys
import os

from tqdm.auto import tqdm
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import numpy as np
import tensorflow as tf

from ctapipe.calib import CameraCalibrator, GainSelector
from ctapipe.core import Tool, QualityQuery
from ctapipe.core.traits import classes_with_traits, Unicode, Float, Integer, Bool
from ctapipe.image import ImageCleaner, ImageModifier, ImageProcessor
from ctapipe.image.extractor import ImageExtractor
from ctapipe.reco.reconstructor import StereoQualityQuery
from ctapipe.instrument import get_atmosphere_profile_functions
from ctapipe.image import dilate

from ctapipe.io import (
    DataLevel,
    EventSource,
    SimTelEventSource,
    metadata,
)
from ctapipe.coordinates import (
    CameraFrame, 
    NominalFrame, 
    GroundFrame, 
    TiltedGroundFrame,
)

from ctapipe.utils import EventTypeFilter
from astropy.time import Time

from sklearn.model_selection import train_test_split

from CNN_template_trainer.utilities import *
import keras

COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
]

from art import tprint

__all__ = ["ProcessorTool"]


class TemplateFitter(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "template-fitter"
    examples = """
    To process data with all default values:
    > template-fitter --input events.simtel.gz --output events.dl1.h5 --progress
    Or use an external configuration file, where you can specify all options:
    > template-fitter --config stage1_config.json --progress
    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    input_files = Unicode(
        default_value=".", help="list of input files"
    ).tag(config=True)

    output_file = Unicode(
        default_value=".", help="base output file name"
    ).tag(config=True)

    reweight_index = Float(
        default_value=0,
        help = " ",
    ).tag(config=True)

    reweight_energy = Float(
        default_value=10,
        help = " ",
    ).tag(config=True)

    atmosphere = Unicode(
        default_value="paranal",
        help = "Profile to use for atmospheric density",
    ).tag(config=True)

    save_images = Unicode(
        default_value=" ",
        help = "Save input images to intermediate file",
    ).tag(config=True)

    load_images = Bool(
        default_value=False,
        help="Load input images from intermediate save file"
    ).tag(config=True)

    filters = Integer(
        default_value=50,
        help = "Number of filters to use in CNN",
    ).tag(config=True)

    layers = Integer(
        default_value=14,
        help = "Number of filters to use in CNN",
    ).tag(config=True)

    image_number = Integer(
        default_value=0,
        help = "Maximum number of images to use for training set ()",
    ).tag(config=True)

    camera_type = Unicode(
        default_value="NectarCam",
        help = "Camera type for templates",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TemplateFitter.input_files",
        ("o", "output"): "TemplateFitter.output_file",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "atmosphere": "TemplateFitter.atmosphere",
        "reweight-index": "TemplateFitter.reweight_index",
        "reweight-energy": "TemplateFitter.reweight_energy",
        "save-images": "TemplateFitter.save_images",
        "load-images": "TemplateFitter.load_images",
        "camera-type": "TemplateFitter.camera_type",
        "layers": "TemplateFitter.layers",
        "filters": "TemplateFitter.filters",
        "image-number": "TemplateFitter.image_number",
    }

    classes = (
        [
            CameraCalibrator,
            ImageProcessor,
            metadata.Instrument,
            metadata.Contact,
        ]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(StereoQualityQuery)        
        + classes_with_traits(ImageModifier)
        + classes_with_traits(EventTypeFilter)

    )


    def setup(self):

        if self.load_images:
            return None
        # setup components:
        self.input_file_list = self.input_files.split(",")

        self.focal_length_choice='EFFECTIVE'
        
        try:
             self.event_source = EventSource(input_url=self.input_file_list[0], parent=self, 
                        focal_length_choice=self.focal_length_choice)
        except RuntimeError:
            print("Effective Focal length not availible, defaulting to equivelent")
            self.focal_length_choice='EQUIVALENT'
            self.event_source = EventSource(input_url=self.input_file_list[0], parent=self, 
                    focal_length_choice=self.focal_length_choice)

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide either R1 or DL0 or DL1A data"
                ", %s provides only %s",
                self.name,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.event_type_filter = EventTypeFilter(parent=self)
        self.check_parameters = StereoQualityQuery(parent=self) 
        
        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input Simulation file are invalid)."
            )

        _ = get_atmosphere_profile_functions(
            self.atmosphere, with_units=False
        )
        self.thickness_profile, altitude_profile = _

        # We need this dummy time for coord conversions later
        self.dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')
        self.point, self.tilt_tel = None, None

        self.count_total = 0
        self.image_count = 0
        self.inputs, self.amplitude, self.time = [], [], []

    def start(self):
        """
        Process events
        """
        if self.load_images:
            return None
        
        self.event_source.subarray.info(printer=self.log.info)

        for input_file in self.input_file_list:
            if not os.path.exists(input_file):
                continue

            self.event_source = EventSource(input_url=input_file, parent=self, 
                focal_length_choice=self.focal_length_choice)
            self.point, self.xmax_scale, self.tilt_tel = None, None, None

            for event in tqdm(
                self.event_source,
                desc=self.event_source.__class__.__name__,
                total=self.event_source.max_events,
                unit="events"
                ):

                self.log.debug("Processessing event_id=%s", event.index.event_id)
                self.calibrate(event)
                self.process_images(event)

                self.read_template(event)
                if self.image_count>self.image_number-1 and self.image_number>0:
                    print( self.image_number, "images reached")
                    self.event_source.close()
                    return
            # Not sure what else to do here...
           #obs_ids = self.event_source.simulation_config.keys()

            self.event_source.close()

    def finish(self):
        """
        Last steps after processing events.
        """

        if self.load_images:
            with np.load(self.input_files) as data:
                inputs= data['inputs']
                amplitude = data['amplitude']
                time = data['time']

        else:
            inputs = np.array(self.inputs)
            amplitude = np.array(self.amplitude)
            time = np.array(self.time)

        if self.save_images != " ":
            np.savez_compressed(self.save_images, inputs=inputs, amplitude=amplitude, time=time)
            
        inputs[:,2:] = np.log10(inputs[:,2:])
        inputs = inputs[:,:-1]
        inputs = np.nan_to_num(inputs, posinf=0, neginf=0)

        inputs, inputs_shuffle, amplitude, amplitude_shuffle, time, time_shuffle = train_test_split(inputs, amplitude, time, test_size=0.5)
        np.random.shuffle(amplitude_shuffle)
        np.random.shuffle(time_shuffle)   

        target = np.ones_like(amplitude)
        target_shuffle = np.zeros_like(amplitude_shuffle)

        print(inputs.shape, amplitude.shape)
        inputs = np.hstack((inputs, np.expand_dims(amplitude, axis=1)))
        inputs_shuffle = np.hstack((inputs_shuffle, np.expand_dims(amplitude_shuffle, axis=1)))

        target = np.concatenate((target, target_shuffle))
        inputs = np.concatenate((inputs, inputs_shuffle))

        inputs, inputs_test, target, target_test = train_test_split(inputs, target, test_size=0.1)

        model = self.generate_templates(inputs, target)
        model.save(self.output_file)

    def read_template(self, event):
        """_summary_

        Args:
            event (_type_): _description_
        """

        # Store simulated event energy
        energy = event.simulation.shower.energy

        weight_factor = (energy/(self.reweight_energy * u.TeV))**self.reweight_index
        if weight_factor != 1.:
            random_value = np.random.rand()
            #print(weight_factor, random_value)
            if random_value > weight_factor:
                #print("reject")
                return

        # When calculating alt we have to account for the case when it is rounded
        # above 90 deg
        alt_evt = event.simulation.shower.alt
        if alt_evt > 90 * u.deg:
            alt_evt = 90*u.deg

        # Get the pointing direction and telescope positions of this run
        if self.point is None:
            alt = event.pointing.array_altitude
            if alt > 90 * u.deg:
                alt = 90*u.deg

            self.point = SkyCoord(alt=alt, az=event.pointing.array_azimuth,
                    frame=AltAz(obstime=self.dummy_time))

            grd_tel = self.event_source.subarray.tel_coords
            # Convert to tilted system
            self.tilt_tel = grd_tel.transform_to(
                TiltedGroundFrame(pointing_direction=self.point))

        # Create coordinate objects for source position
        src = SkyCoord(alt=event.simulation.shower.alt.value * u.rad, 
                        az=event.simulation.shower.az.value * u.rad,
                        frame=AltAz(obstime=self.dummy_time))

        zen = 90 - event.simulation.shower.alt.to(u.deg).value
        # Store simulated Xmax
        mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

        # And transform into nominal system (where we store our templates)
        source_direction = src.transform_to(NominalFrame(origin=self.point))

        # Calculate core position in tilted system
        grd_core_true = SkyCoord(x=np.asarray(event.simulation.shower.core_x) * u.m,
                                    y=np.asarray(event.simulation.shower.core_y) * u.m,
                                    z=np.asarray(0) * u.m, frame=GroundFrame())

        tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
            pointing_direction=self.point))

        # Loop over triggered telescopes
        for tel_id, dl1 in event.dl1.tel.items():
            #  Get pixel signal

            if self.event_source.subarray.tel[tel_id].camera.name != self.camera_type:
                continue
            if np.invert(self.check_parameters(parameters=dl1.parameters).all()):
                continue
            mask = event.dl1.tel[tel_id].image_mask
            geom = self.event_source.subarray.tel[tel_id].camera.geometry

            for i in range(4):
                mask = dilate(geom, mask)
            pmt_signal = dl1.image[mask]

            # Get pixel coordinates and convert to the nominal system
            fl = self.event_source.subarray.tel[tel_id].optics.effective_focal_length

            camera_coord = SkyCoord(x=geom.pix_x, y=geom.pix_y,
                                    frame=CameraFrame(focal_length=fl, telescope_pointing=self.point))

            nom_coord = camera_coord.transform_to(
                NominalFrame(origin=self.point))

            x = nom_coord.fov_lon.to(u.deg)[mask]
            y = nom_coord.fov_lat.to(u.deg)[mask]

            tel_num = np.argwhere(self.event_source.subarray.tel_ids == tel_id)
            # Calculate expected rotation angle of the image
            phi = np.arctan2((self.tilt_tel.y[tel_num] - tilt_core_true.y),
                                (self.tilt_tel.x[tel_num] - tilt_core_true.x)) + \
                    90 * u.deg

            # And the impact distance of the shower
            impact = np.sqrt(np.power(self.tilt_tel.x[tel_num] - tilt_core_true.x, 2) +
                                np.power(self.tilt_tel.y[tel_num] - tilt_core_true.y, 2)). \
                to(u.m).value
            
            # now rotate and translate our images such that they lie on top of one
            # another
            x, y = rotate_translate(x, y,
                                    source_direction.fov_lon,
                                    source_direction.fov_lat, phi)
            x *= -1 # Reverse x axis to fit HESS convention
            x, y = x.ravel(), y.ravel()

            # Store simulated Xmax
            mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))  

            #az = self.point.az.to(u.deg).value
            #zen = 90. - self.point.alt.to(u.deg).value
            h0 =  event.simulation.shower.h_first_int
            x0 = self.thickness_profile(h0)/np.cos(np.deg2rad(zen))

            for i in range(len(x)):
                self.inputs.append((x[i].to(u.deg).value, y[i].to(u.deg).value, energy.value, impact[0][0], mc_xmax, x0))
                self.amplitude.append(pmt_signal[i])
                self.time.append(pmt_signal[i])

            self.image_count += 1

    def generate_templates(self, inputs, target):
        """_summary_

        Args:
            inputs (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        model = create_model((6))
        #impacts = 10**inputs[:, 20,20,2]
        #weight = (impacts/150)**-0.7
        #weight =  np.expand_dims(weight, axis=1)

        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="binary_crossentropy", optimizer=adam, weighted_metrics=[])
#        model.compile(loss="mae", optimizer=adam, weighted_metrics=[])
        print(model.summary())
        # Setup the callbacks, checkpointing, logger and early stopper
        csv_logger = keras.callbacks.CSVLogger('training.log')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                    patience=20, min_lr=0.000001)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./",
            monitor='val_loss',
            mode='min',
            save_best_only=True, cooldown=10)

        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, 
                                                    restore_best_weights=True, start_from_epoch=100)

        model.fit(inputs.astype("float32"), target.astype("float32"), epochs=2000,
                batch_size=10000,shuffle=True, validation_split=0.2,
#                sample_weight = weight,
                callbacks=[csv_logger, reduce_lr,  model_checkpoint_callback, stopping])

        return model
    

def main():
    """run the tool"""
    print("=======================================================================================")
    tprint("Template   Fitter")
    print("=======================================================================================")

    tool = TemplateFitter()
    tool.run()

if __name__ == "__main__":
    main()
