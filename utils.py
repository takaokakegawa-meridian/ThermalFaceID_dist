from senxor.utils import connect_senxor
from senxor.mi48 import MI48


def config_mi48(params: dict) -> MI48:
  """configure mi48 object to register parameters passed in as dictionary
  Args:
    params (dict): mi48 reigster parameters to write in, as a dictionary.
  Returns:
    MI48: detected mi48 object that has been connected, registered parameters written in.
  """
  mi48 = connect_senxor()
  for reg in params['regwrite']:
    mi48.regwrite(*reg)
  if 'sens_factor' in params.keys():
    mi48.set_sens_factor(params['sens_factor'])
  if 'offset_corr' in params.keys():
    mi48.set_sens_factor(params['offset_corr'])
  if 'emissivity' in params.keys():
    mi48.set_sens_factor(params['emissivity'])
  return mi48