import numpy as np

def f(x):
  return np.exp(x)
def df(x):
  return np.exp(x)
def ddf(x):
  return np.exp(x)

def left_rectangles():
  return h * (abs(y[:n-1]).sum())
def left_rectangles_error():
  max_df = np.apply_along_axis(df, 0, x).max()
  return max_df/2*h*(b-a)

def right_rectangles():
  return h * (abs(y[1:]).sum())
def right_rectangles_error():
  max_df = np.apply_along_axis(df, 0, x).max()
  return max_df/2*h*(b-a)

def center_rectangles():
  return h * (np.apply_along_axis(lambda i: f(i - h/2), 0, abs(x[1:n-1])).sum())
def center_rectangles_error():
  max_df = np.apply_along_axis(ddf, 0, x).min()
  return max_df*(h)*(b-a)/24

def trapeze():
  return h * ((y[0] + y[-1])/2 + abs(y[1:n-1]).sum())
def trapeze_error():
  return (b-a)*(f(a)+4*f((a+b)/2)+f(b))/12

def Simpson():
  return (b-a)/6 * (f(a) + 4*f(a+b/2) + f(b))
def Simpson_error():
  max_df4 = b
  return ((b-a)**5)*max_df4/2880
a, b = 0, 1
n = 10000
h = (b-a)/n
x = np.linspace(a, b, n)
y = np.apply_along_axis(f, 0, x)
ans = np.exp(1) - 1
#%%
print(f"n = {n}")

print(f"Интервал: [{a:.2f}, {b:.2f}]")
print(f"Точное значение: {ans}")
print()
print(f"""Метод левых прямоугольников:
ans {left_rectangles():.8f}
er  {abs(ans - left_rectangles()):.3e}
O(h){abs(left_rectangles_error()):.3e}
Метод правых прямоугольников:
ans {right_rectangles():.8f}
er  {abs(ans - right_rectangles()):.3e}
O(h){abs(left_rectangles_error()):.3e}
Метод центральных прямоугольников:
ans {center_rectangles():.8f}
er  {abs(ans - center_rectangles()):.3e}
O(h){np.sqrt(abs(center_rectangles_error())):.3e}
Метод трапеций:
ans {trapeze():.8e}
er  {abs(ans - trapeze()):.3e}
O(h){trapeze_error():.3e}
Метод Симпсона:
er  {Simpson():.8e}
asn {abs(ans - Simpson()):.3e}
O(h){abs(Simpson_error()):.3e}""")