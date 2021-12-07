from distutils.core import setup

setup(name='FTX Client',
      version='0.1',
      description='API Client for FTX',
      author='Austin Steady',
      author_email='asteady23@gmail.com',
      url='http://austinsteady.com',
      packages=['ftx_client'],
      install_requires=['pandas', 'requests', 'simplejson']
     )
