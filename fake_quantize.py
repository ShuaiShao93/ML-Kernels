import math

def fake_quant(inputs):
  min = -7.
  max = 557.
  quant_min_float = -127.
  quant_max_float = 127.
  scale = (max - min) / (quant_max_float - quant_min_float)
  inv_scale = (quant_max_float - quant_min_float) / (max - min)
  print("scale", scale)
  zero_point_from_min = quant_min_float - min / scale

  if (zero_point_from_min < quant_min_float):
    nudged_zero_point = int(quant_min_float)
  elif (zero_point_from_min > quant_max_float):
    nudged_zero_point = int(quant_max_float)
  else:
    nudged_zero_point = int(round(zero_point_from_min))
  print("nudged_zero_point", nudged_zero_point)

  nudged_min = (quant_min_float - nudged_zero_point) * scale
  nudged_max = (quant_max_float - nudged_zero_point) * scale
  print("nudged_min", nudged_min)
  print("nudged_max", nudged_max)

  quant_zero = math.floor(-nudged_min * inv_scale + 0.5)
  print("quant_zero", quant_zero)

  # clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
  clamped = inputs
  clamped_shifted = clamped - nudged_min
  return math.floor(clamped_shifted * inv_scale - quant_zero + 0.5) * scale


print(fake_quant(75.3536453))