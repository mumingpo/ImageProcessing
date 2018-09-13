import sys
import numpy as np
from PIL import Image as img
from scipy.stats import norm

S = {
  'path': None,
  #'path_out':'2.png',
  'mode': 1,
  'offset': 2,
}

def main():
  if S['path'] is None:
    for flags in sys.argv[1:]:
      option, value = flags[:flags.find('=')], flags[flags.find('=')+1:]
      S[option] = int(value) if option not in ('path', 'path_out') else value
    
  with img.open(S['path']) as p:
    p.show()
    y, cb, cr = map(lambda x: np.array(x).astype(np.float), p.convert('YCbCr').split())
  
  ym, cbm, crm = map(lambda x: mapoffset(x, S['offset']), [y, cb, cr])
  
  ps = []
#  if S['mode'] == 0 or S['mode'] == -1:
#    det = v - vr - vd + vrd
#    filter = np.abs(det.copy())
#    filter[filter<10] = 0
#    filter[filter!=0] = 255
#    ps.append(img.fromarray(normalize(filter, S['offset'])))
  if S['mode'] == 1 or S['mode'] == -1:
    ymmean, cbmmean, crmmean = map(lambda x: sum([sum(y) for y in x])/(S['offset']**2), [ym, cbm, crm])
    ymvar, cbmvar, crmvar = map(lambda x, y: sum([sum([(x[i][j]-y)**2 for i in range(S['offset'])]) for j in range(S['offset'])])/(S['offset']**2), [ym, cbm, crm], [ymmean, cbmmean, crmmean])
    sig = sum([ymvar, cbmvar, crmvar]) ** (1/5)
    ps.append(img.fromarray(normalize(sig, S['offset'])))
#  if S['mode'] == 2 or S['mode'] == -1:
#    fv, fvr, fvd, fvrd = map(np.fft.fft2, (v, vr, vd, vrd))
#    fdet = fv - fvr - fvd + fvrd
#    det = np.abs(np.fft.ifft2(fdet))
#    ps.append(img.fromarray(normalize(det)))
#  if S['mode'] == 3 or S['mode'] == -1:
#    fv, fvr, fvd, fvrd = map(np.fft.fft2, (v, vr, vd, vrd))
#    fdet = fv - fvr - fvd + fvrd
#    det = np.abs(np.fft.ifft2(fdet))
#    logdet = np.log(det, out=np.zeros_like(det), where=det>0)
#    logdet[logdet<0] = 0
#    ps.append(img.fromarray(normalize(logdet)))
  if S['mode'] == 4 or S['mode'] == -1:
    ymmean, cbmmean, crmmean = map(lambda x: sum([sum(y) for y in x])/(S['offset']**2), [ym, cbm, crm])
    ymsig, cbmsig, crmsig = map(lambda x, y: (sum([sum([(x[i][j]-y)**2 for i in range(S['offset'])]) for j in range(S['offset'])])/(S['offset']**2))**0.5, [ym, cbm, crm], [ymmean, cbmmean, crmmean])
    ymprob, cbmprob, crmprob = map(lambda x, y, z: norm.cdf(np.divide(abs(x - y), z, out=np.zeros_like(z), where=z!=0)), [y[:1-S['offset'],:1-S['offset']], cb[:1-S['offset'],:1-S['offset']], cr[:1-S['offset'],:1-S['offset']]], [ymmean, cbmmean, crmmean], [ymsig, cbmsig, crmsig])
    prob = 1 - (1 - ymprob) * (1 - cbmprob) * (1 - crmprob)
    ps.append(img.fromarray(normalize(prob, S['offset'])))
    
  for i, j in enumerate(ps):
    j.show()
    j.save('{}_ed_{}.png'.format(S['path'][:S['path'].find('.')], i))
  
def normalize(inputarray, offset=2):
  array = inputarray.copy()[:1-offset,:1-offset]
  array -= array.min()
  array /= array.max()
  array *= 255
  array = array.astype(np.uint8)
  array = 255 - array
  return array
  
def mapoffset(a:np.array, offset = 2):
  a = a.copy()[:1 - offset, :1 - offset]
  m = [[np.zeros_like(a) for _ in range(offset)] for __ in range(offset)]
  for i in range(offset):
    for j in range(offset):
      if i == 0:
        ii = a.shape[0]
      else:
        ii = -i
      if j == 0:
        ji = a.shape[1]
      else:
        ji = -j
      m[i][j][:ii,:ji] = a[i:,j:]
  return m
  
if __name__ == '__main__':
  main()