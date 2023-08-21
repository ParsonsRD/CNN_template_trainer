from setuptools import setup

setup(
    name='CNN_template_trainer',
    version='1.0',
    packages=['CNN_template_trainer'],
    package_dir={'CNN_template_trainer': 'CNN_template_trainer'},
    #package_data={'CNN_template_trainer': ["configs/array_trigger_temp.dat",
    #                                   "configs/cta-temp_run.sh",
    #                                   "configs/run_sim_template",
    #                                   "configs/simtel_template.cfg"],
    #              'CNN_template_trainer': ["data/gamma_HESS_example.simhess.gz"]},
    #include_package_data=True,
    url='',
    license='',
    author='parsonsrd',
    author_email='',
    description='Creation tools for building ImPACT templates for ctapipe',
    entry_points = {'console_scripts': ['template-fitter = CNN_template_trainer.template_fitter:main']},
)
