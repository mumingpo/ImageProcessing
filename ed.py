import sys
import numpy as np
from PIL import Image as img

S = {
  'path': 'u.png',
  #'path_out':'2.png',
  'mode': 1,
}

def main():
  for flags in sys.argv[1:]:
    option, value = flags[:flags.find('=')], flags[flags.find('=')+1:]
    S[option] = int(value) if option not in ('path', 'path_out') else value
    
  with img.open(S['path']) as p:
    y = p.convert('YCbCr').split()[0]
    v = np.array(y).astype(np.float)
  
  vr, vd, vrd = (np.zeros_like(v) for _ in range(3))
  vr[:-1,:] = v[1:,:]
  vd[:,:-1] = v[:,1:]
  vrd[:-1,:-1] = v[1:,1:]
  
  ps = [y]
  if S['mode'] == 0 or S['mode'] == -1:
    det = v - vr - vd + vrd
    filter = np.abs(det.copy())
    filter[filter<10] = 0
    filter[filter!=0] = 255
    ps.append(img.fromarray(normalize(filter)))
  if S['mode'] == 1 or S['mode'] == -1:
    mean = (v+vr+vd+vrd)/4
    sig = (sum(map(lambda x:(x-mean)**2, [v,vr,vd,vrd]))/4)**0.5
    ps.append(img.fromarray(normalize(sig)))
  if S['mode'] == 2 or S['mode'] == -1:
    fv, fvr, fvd, fvrd = map(np.fft.fft2, (v, vr, vd, vrd))
    fdet = fv - fvr - fvd + fvrd
    det = np.abs(np.fft.ifft2(fdet))
    ps.append(img.fromarray(normalize(det)))
  if S['mode'] == 3 or S['mode'] == -1:
    fv, fvr, fvd, fvrd = map(np.fft.fft2, (v, vr, vd, vrd))
    fdet = fv - fvr - fvd + fvrd
    det = np.abs(np.fft.ifft2(fdet))
    logdet = np.log(det, out=np.zeros_like(det), where=det>0)
    logdet[logdet<0] = 0
    ps.append(img.fromarray(normalize(logdet)))
    
  for i, j in enumerate(ps):
    j.show()
    j.save('{}_ed_{}.png'.format(S['path'][:S['path'].find('.')], i))
  
def normalize(inputarray):
  array = inputarray.copy()[:-1,:-1]
  array -= array.min()
  array /= array.max()
  array *= 255
  array = array.astype(np.uint8)
  array = 255 - array
  return array[:-1,:-1]
  
def mapoffset(a:np.array, offset = 2, scale = 1):
  a = a.copy()[::scale,::scale]
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