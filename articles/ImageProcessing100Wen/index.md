---
title: '图像处理100问'
publish_time:'2020-10-01'
updates:
hidden: false
---

### 1.通道交换

cv2.imread( ) 读入的图片系数是按{BGR}顺序排列的

```python
import cv2
img = cv2.imread("imori.jpg")
red = img[:, :, 2].copy()  #取出红色通道
```

```python
# Read image
img = cv2.imread("../imori.jpg")
# BGR -> RGB
img = BGR2RGB(img)
```

### 2.灰度化

灰度是一种图像亮度的表示方法，通过下式计算：

$$
Y = 0.2126\  R + 0.7152\  G + 0.0722\  B
$$

```python
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out
```

### 3.二值化

$$
y=
\begin{cases}
0& (\text{if}\quad y < 128) \\
255& (\text{else})
\end{cases}
$$

```python
#先要进行灰度化变为1通道，然后进行二值化
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out

# binalization
def binarization(img, th=128):
    img[img < th] = 0
    img[img >= th] = 255
    return img

# Read image
img = cv2.imread("../imori.jpg").astype(np.float32)

# Grayscale
out = BGR2GRAY(img)

# Binarization
out = binarization(out)
```

### 4.大津二值化算法

大津算法，也被称作最大类间方差法，是一种可以自动确定二值化中阈值的算法。
从**类内方差**和**类间方差**的比值计算得来：

- 小于阈值$t$的类记作$0$，大于阈值$t$的类记作$1$；
- $w_0$和$w_1$是被阈值$t$分开的两个类中的像素数占总像素数的比率（满足$w_0+w_1=1$）；
- ${S_0}^2$， ${S_1}^2$是这两个类中像素值的方差；
- $M_0$，$M_1$是这两个类的像素值的平均值；

即：

* 类内方差：${S_w}^2=w_0\ {S_0}^2+w_1\  {S_1}^2$
* 类间方差：${S_b}^2 = w_0 \  (M_0 - M_t)^2 + w_1\ (M_1 - M_t)^2 = w_0\  w_1\  (M_0 - M_1) ^2$
* 图像所有像素的方差：${S_t}^2 = {S_w}^2 + {S_b}^2 = \text{常数}$

根据以上的式子，我们用以下的式子计算分离度$X$：[^1]

也就是说：

$$
\arg\max\limits_{t}\ X=\arg\max\limits_{t}\ {S_b}^2
$$

换言之，如果使${S_b}^2={w_0}\ {w_1}\ (M_0 - M_1)^2$最大，就可以得到最好的二值化阈值$t$。

```Python
#先进行灰度化
#二值化
def otsu_binarization(img, th=128):
	max_sigma = 0
	max_t = 0

	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]  ##取出小于阈值的像素值
		m0 = np.mean(v0) if len(v0) > 0 else 0.  #求小于阈值的平均M
		w0 = len(v0) / (H * W)  #求小于阈值的点占总像素的比例
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:  ##记录最大值
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255
	return out
```

### 5.$\text{HSV}$变换

$\text{HSV}$即使用**色相（Hue）、饱和度（Saturation）、明度（Value）**来表示色彩的一种方式。

| - 色相：将颜色使用$0^{\circ}$到$360^{\circ}$表示，就是平常所说的颜色名称，如红色、蓝色。色相与数值按下表对应： | 红           | 黄            | 绿            | 青色          | 蓝色          | 品红          | 红 |
| -------------------------------------------------------------------------------------------------------------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| $0^{\circ}$                                                                                                    | $60^{\circ}$ | $120^{\circ}$ | $180^{\circ}$ | $240^{\circ}$ | $300^{\circ}$ | $360^{\circ}$ |

- 饱和度：是指色彩的纯度，饱和度越低则颜色越黯淡（$0\leq S < 1$）；
- 明度：即颜色的明暗程度。数值越高越接近白色，数值越低越接近黑色（$0\leq V < 1$）；

从$\text{RGB}$色彩表示转换到$\text{HSV}$色彩表示通过以下方式计算：

$\text{RGB}$的取值范围为$[0, 1]$，令：

$$
\text{Max}=\max(R,G,B)\\
\text{Min}=\min(R,G,B)
$$

色相：

$$
H=\begin{cases}
0&(\text{if}\ \text{Min}=\text{Max})\\
60\  \frac{G-R}{\text{Max}-\text{Min}}+60&(\text{if}\ \text{Min}=B)\\
60\  \frac{B-G}{\text{Max}-\text{Min}}+180&(\text{if}\ \text{Min}=R)\\
60\  \frac{R-B}{\text{Max}-\text{Min}}+300&(\text{if}\ \text{Min}=G)
\end{cases}
$$

饱和度：

$$
S=\text{Max}-\text{Min}
$$

明度：

$$
V=\text{Max}
$$

从$\text{HSV}$色彩表示转换到$\text{RGB}$色彩表示通过以下方式计算：

$$
C = S\\
H' = \frac{H}{60}\\
X = C\  (1 - |H' \mod 2 - 1|)\\
(R,G,B)=(V-C)\ (1,1,1)+\begin{cases}
(0, 0, 0)&  (\text{if H is undefined})\\
(C, X, 0)&  (\text{if}\quad 0 \leq H' < 1)\\
(X, C, 0)&  (\text{if}\quad 1 \leq H' < 2)\\
(0, C, X)&  (\text{if}\quad 2 \leq H' < 3)\\
(0, X, C)&  (\text{if}\quad 3 \leq H' < 4)\\
(X, 0, C)&  (\text{if}\quad 4 \leq H' < 5)\\
(C, 0, X)&  (\text{if}\quad 5 \leq H' < 6)
\end{cases}
$$

例：请将色相反转（色相值加$180$），然后再用$\text{RGB}$色彩空间表示图片。

```Python
import cv2
import numpy as np

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.# 使RGB的取值范围变为0到1

	hsv = np.zeros_like(img, dtype=np.float32)    #生成与img相同形状的数组

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2) #确实最小值在数组中的位置(哪一页)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()

	return hsv


def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()  #axis=0时表示取每一列的最大值
	min_v = np.min(img, axis=2).copy()  #axis=1时表示取每一行的最小值
										#axis=2时表示取每一页的最小值
	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)

	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img)

# Transpose Hue
hsv[..., 0] = (hsv[..., 0] + 180) % 360

# HSV > RGB
out = HSV2BGR(img, hsv)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

对axis的理解
![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E7%90%86%E8%A7%A3axis.png)

当axis=0的时候表示取***上下两维***的最大值：

```Python
np.max(z, axis=0)
array([[12, 13, 14, 15],     
       [16, 17, 18, 19],       
       [20, 21, 22, 23]])  
```

当axis=1的时候表示取上下两维列的最大值：

```Python
np.max(z, axis=1)          
array([[ 8,  9, 10, 11],      
       [20, 21, 22, 23]])
```

axis=2的时候表示取上下两维行的最大值：

```Python
np.max(z, axis=2)   
array([[ 3,  7, 11],      
       [15, 19, 23]]) 
```

### 6.减色处理

我们将图像的值由$256^3$压缩至$4^3$，即将$\text{RGB}$的值只取$\{32, 96, 160, 224\}$。这被称作色彩量化。色彩的值按照下面的方式定义：

$$
\text{val}=
\begin{cases}
32& (0 \leq \text{var} <  64)\\
96& (64\leq \text{var}<128)\\
160&(128\leq \text{var}<192)\\
224&(192\leq \text{var}<256)
\end{cases}
$$

```Python
def dicrease_color(img):
	out = img.copy()

	out = out // 64 * 64 + 32    ##//是除后向下取整

	return out
```

### 7.平均池化

将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值。

把大小为$128\times128$的 `imori.jpg`使用$8\times8$的网格做平均池化。

```Python
def average_pooling(img, G=8):
    out = img.copy()
    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)
  
    return out
```

### 8.最大池化

网格内的值不取平均值，而是取网格内的最大值进行池化操作。

```Python
def max_pooling(img, G=8):
    # Max Pooling
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])

    return out
```

### 9.高斯滤波

使用高斯滤波器（$3\times3$大小，标准差$\sigma=1.3$）来进行降噪处理
高斯滤波器是一种可以使图像**平滑**的滤波器，用于去除**噪声**。可用于去除噪声的滤波器还有中值滤波器（参见问题十），平滑滤波器（参见问题十一）、LoG滤波器（参见问题十九）。
高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为**卷积核（kernel）**或者**滤波器（filter）**。

但是，由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补$0$。这种方法称作**Zero Padding**。并且权值$g$（卷积核）要进行[归一化操作](https://blog.csdn.net/lz0499/article/details/54015150)（$\sum\ g = 1$）。

按下面的高斯分布公式计算权值：

$$
g(x,y,\sigma)=\frac{1}{2\  \pi\ \sigma^2}\  e^{-\frac{x^2+y^2}{2\  \sigma^2}}
$$

标准差$\sigma=1.3$的$8-$近邻高斯滤波器如下：

$$
K=\frac{1}{16}\  \left[
 \begin{matrix}
   1 & 2 & 1 \\
   2 & 4 & 2 \\
   1 & 2 & 1
  \end{matrix}
  \right]
$$

```Python
def gaussian_filter(img, K_size=3, sigma=1.3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		img = np.expand_dims(img, axis=-1)
		H, W, C = img.shape

	## Zero padding，边缘补0
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

	## prepare Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad + K_size):
		for y in range(-pad, -pad + K_size):
			K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
	K /= (2 * np.pi * sigma * sigma)
	K /= K.sum()

	tmp = out.copy()

	# filtering
	for y in range(H):
		for x in range(W):
			for c in range(C):
				out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

	out = np.clip(out, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out
```

### 10.中值滤波

使用中值滤波器（$3\times3$大小）进行降噪处理吧

中值滤波器是一种可以使图像平滑的滤波器。这种滤波器用滤波器范围内（在这里是$3\times3$）像素点的中值进行滤波，请在这里也采用Zero Padding。

```Python
def median_filter(img, K_size=3):
    H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.median(tmp[y:y+K_size, x:x+K_size, c])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out
```

### 11.均值滤波器

使用$3\times3$的均值滤波器来进行滤波吧！

```Python
def mean_filter(img, K_size=3):
    H, W, C = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.mean(tmp[y: y + K_size, x: x + K_size, c])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out
```

### 12.Motion Filter

使用$3\times3$的Motion Filter来进行滤波吧。

Motion Filter取对角线方向的像素的平均值，像下式这样定义：

$$
\left[
\begin{matrix}
\frac{1}{3}&0&0\\
0&\frac{1}{3}&0\\
0  & 0&  \frac{1}{3}
\end{matrix}
\right]
$$

```Python
def motion_filter(img, K_size=3):
    H, W, C = img.shape

    # Kernel
    K = np.diag( [1] * K_size ).astype(np.float)
    K /= K_size

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out
```

### 13.MAX-MIN滤波器

MAX-MIN滤波器使用网格内像素的最大值和最小值的差值对网格内像素重新赋值。通常用于**边缘检测**。
边缘检测用于检测图像中的线。像这样提取图像中的信息的操作被称为**特征提取**。
边缘检测通常在灰度图像上进行。

```Python
#先要进行灰度化
#然后再进行MAX-MIN滤波器
def max_min_filter(img, K_size=3):
    H, W = img.shape

    # Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.max(tmp[y: y + K_size, x: x + K_size]) - \
                np.min(tmp[y: y + K_size, x: x + K_size])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out
```

### 14.差分滤波器（Differential Filter）

使用$3\times3$的差分滤波器来进行滤波吧。

差分滤波器对图像亮度急剧变化的边缘有提取效果，可以获得邻接像素的差值。

纵向：

$$
K=\left[
\begin{matrix}
0&-1&0\\
0&1&0\\
0&0&0
\end{matrix}
\right]
$$

横向：

$$
K=\left[
\begin{matrix}
0&0&0\\
-1&1&0\\
0&0&0
\end{matrix}
\right]
$$

```Python
#先进行灰度化
#然后进行差分滤波
def different_filter(img, K_size=3):
	H, W, C = img.shape

	# Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
	tmp = out.copy()

	out_v = out.copy()
	out_h = out.copy()

	# vertical kernel
	Kv = [[0., -1., 0.],[0., 1., 0.],[0., 0., 0.]]
	# horizontal kernel
	Kh = [[0., 0., 0.],[-1., 1., 0.], [0., 0., 0.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
			out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

	out_v = np.clip(out_v, 0, 255)
	out_h = np.clip(out_h, 0, 255)

	out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out_v, out_h
```

### 15.Sobel滤波器

Sobel滤波器可以提取特定方向（纵向或横向）的边缘，滤波器按下式定义：

纵向：

$$
K=\left[
\begin{matrix}
1&2&1\\
0&0&0\\
-1&-2&-1
\end{matrix}
\right]
$$

横向：

$$
K=\left[
\begin{matrix}
1&0&-1\\
2&0&-2\\
1&0&-1
\end{matrix}
\right]
$$

```Python
#先要进行灰度化
#再进行滤波
def sobel_filter(img, K_size=3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		img = np.expand_dims(img, axis=-1)
		H, W, C = img.shape

	# Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
	tmp = out.copy()

	out_v = out.copy()
	out_h = out.copy()

	## Sobel vertical
	Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
	## Sobel horizontal
	Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
			out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

	out_v = np.clip(out_v, 0, 255)
	out_h = np.clip(out_h, 0, 255)

	out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out_v, out_h
```

### 16.Prewitt滤波器

Prewitt滤波器是用于边缘检测的一种滤波器，使用下式定义：
纵向：

$$
K=\left[
\begin{matrix}
-1&-1&-1\\
0&0&0\\
1&1&1
\end{matrix}
\right]
$$

横向：

$$
K=\left[
\begin{matrix}
-1&0&-1\\
-1&0&1\\
-1&0&1
\end{matrix}
\right]
$$

### 17.Laplacian滤波器

Laplacian滤波器是对图像亮度进行二次微分从而检测边缘的滤波器。由于数字图像是离散的，$x$方向和$y$方向的一次微分分别按照以下式子计算：

$$
I_x(x,y)=\frac{I(x+1,y)-I(x,y)}{(x+1)-x}=I(x+1,y)-I(x,y)\\
I_y(x,y) =\frac{I(x, y+1) - I(x,y)}{(y+1)-y}= I(x, y+1) - I(x,y)
$$

因此二次微分按照以下式子计算：

$$
\begin{align*}
&I_{xx}(x,y) \\
=& \frac{I_x(x,y) - I_x(x-1,y)}{(x+1)-x} \\
=& I_x(x,y) - I_x(x-1,y)\\
         =&[I(x+1, y) - I(x,y)] - [I(x, y) - I(x-1,y)]\\
         =& I(x+1,y) - 2\  I(x,y) + I(x-1,y)
\end{align*}
$$

同理：

$$
I_{yy}(x,y)=I(x,y+1)-2\  I(x,y)+I(x,y-1)
$$

特此，Laplacian 表达式如下：

$$
\begin{align*}
&\nabla^2\ I(x,y)\\
=&I_{xx}(x,y)+I_{yy}(x,y)\\
=&I(x-1,y) + I(x,y-1) - 4 * I(x,y) + I(x+1,y) + I(x,y+1)
\end{align*}
$$

如果把这个式子表示为卷积核是下面这样的：

$$
K=
\left[
\begin{matrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{matrix}
\right]
$$

```Python
def laplacian_filter(img, K_size=3):
	H, W, C = img.shape

	# zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
	tmp = out.copy()

	# laplacian kernle
	K = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out[pad + y, pad + x] = np.sum(K * (tmp[y: y + K_size, x: x + K_size]))

	out = np.clip(out, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out
```

### 18.Emboss滤波器

Emboss滤波器可以使物体轮廓更加清晰，按照以下式子定义：

$$
K=
\left[
\begin{matrix}
-2&-1&0\\
-1&1&1\\
0&1&2
\end{matrix}
\right]
$$

### 19.LoG滤波器

LoG即高斯-拉普拉斯（Laplacian of Gaussian）的缩写，使用高斯滤波器使图像平滑化之后再使用拉普拉斯滤波器使图像的轮廓更加清晰。

为了防止拉普拉斯滤波器计算二次微分会使得图像噪声更加明显，所以我们首先使用高斯滤波器来抑制噪声。

 LoG  滤波器使用以下式子定义：

$$
\text{LoG}(x,y)=\frac{x^2 + y^2 - s^2}{2 \  \pi \  s^6} \  e^{-\frac{x^2+y^2}{2\  s^2}}
$$

```Python
def LoG_filter(img, K_size=5, sigma=3):
	H, W, C = img.shape

	# zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
	tmp = out.copy()

	# LoG Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad + K_size):
		for y in range(-pad, -pad + K_size):
			K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
	K /= (2 * np.pi * (sigma ** 6))
	K /= K.sum()

	# filtering
	for y in range(H):
		for x in range(W):
			out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

	out = np.clip(out, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out
```

### 20.直方图

使用 `Matplotlib`来绘制的直方图
直方图显示了不同数值的像素出现的次数。在 `Matplotlib`中有 `hist()`函数提供绘制直方图的接口。

```Python
# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)

# Display histogram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out.png")
plt.show()
```

### 21.直方图归一化

有时直方图会偏向一边。

比如说，数据集中在$0$处（左侧）的图像全体会偏暗，数据集中在$255$处（右侧）的图像会偏亮。

如果直方图有所偏向，那么其**[动态范围（ dynamic range ）](https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%8C%83%E5%9B%B4)**就会较低。

为了使人能更清楚地看见图片，让直方图归一化、平坦化是十分必要的。

这种归一化直方图的操作被称作**灰度变换（Grayscale Transformation）**。像素点取值范围从$[c,d]$转换到$[a,b]$的过程由下式定义。这回我们将 `imori_dark.jpg`的灰度扩展到$[0, 255]$范围：

$$
x_{out}=
\begin{cases}
a& (\text{if}\quad x_{in}<c)\\
\frac{b-a}{d-c}\ (x_{in}-c)+a&(\text{else if}\quad c\leq x_{in}<d)\\
b&(\text{else})
\end{cases}
$$

```Python
def hist_normalization(img, a=0, b=255):
	# get max and min
	c = img.min()
	d = img.max()

	out = img.copy()

	# normalization
	out = (b-a) / (d - c) * (out - c) + a
	out[out < a] = a
	out[out > b] = b
	out = out.astype(np.uint8)

	return out
```

### 22.直方图操作

可以改变均值和标准差
这里并不是变更直方图的动态范围，而是让直方图变得平坦。

可以使用下式将平均值为$m$标准差为$s$的直方图变成平均值为$m_0$标准差为$s_0$的直方图：

$$
x_{out}=\frac{s_0}{s}\  (x_{in}-m)+m_0
$$

让直方图的平均值$m_0=128$，标准差$s_0=52$吧！

```Python
def hist_mani(img, m0=128, s0=52):
	m = np.mean(img)
	s = np.std(img)

	out = img.copy()

	# normalize
	out = s0 / s * (out - m) + m0
	out[out < 0] = 0
	out[out > 255] = 255
	out = out.astype(np.uint8)

	return out
```

### 23.直方图均衡化

来让均匀化直方图吧！

直方图均衡化是使直方图变得平坦的操作，是不需要计算上面的问题中的平均值、标准差等数据使直方图的值变得均衡的操作。

均衡化操作由以下式子定义。$S$是总的像素数；$Z_{max}$是像素点的最大取值（在这里是$255$）；$h(z)$表示取值为$z$的累积分布函数：

$$
Z' = \frac{Z_{max}}{S} \  \sum\limits_{i=0}^z\ h(i)
$$

这个公式就是将像素点的个数与色阶挂钩 : 某一像素值及小于其像素值的总数量占所有像素的比例要与其像素值占总像素值的比值要一样

```Python
def hist_equal(img, z_max=255):
	H, W, C = img.shape
	S = H * W * C * 1.

	out = img.copy()

	sum_h = 0.

	for i in range(1, 255):
		ind = np.where(img == i) #取出某一值的像素点位置
		sum_h += len(img[ind])  #小于等于该值像素点的个数
		z_prime = z_max / S * sum_h
		out[ind] = z_prime  #改变这些点的像素值

	out = out.astype(np.uint8)

	return out
```

### 24.伽玛校正（Gamma Correction）

对图像进行伽马校正（$c=1$，$g=2.2$）

伽马校正用来对照相机等电子设备传感器的非线性光电转换特性进行校正。如果图像原样显示在显示器等上，画面就会显得很暗。伽马校正通过预先增大 RGB 的值来排除显示器的影响，达到对图像修正的目的。

由于下式引起非线性变换，在该式中，$x$被归一化，限定在$[0,1]$范围内。$c$是常数，$g$为伽马变量（通常取$2.2$）：

$$
x' = c\  {I_{in}}^ g
$$

因此，使用下面的式子进行伽马校正：

$$
I_{out} ={\frac{1}{c}\  I_{in}} ^ {\frac{1}{g}}
$$

对图像进行伽马校正（$c=1$，$g=2.2$）

```Python
def gamma_correction(img, c=1, g=2.2):
	out = img.copy()
	out /= 255.
	out = (1/c * out) ** (1/g)

	out *= 255
	out = out.astype(np.uint8)

	return out
```

### 25.最邻近差值（ Nearest-neighbor Interpolation ）

使用最邻近插值将图像放大$1.5$倍吧！

最近邻插值在图像放大时补充的像素取最临近的像素的值。由于方法简单，所以处理速度很快，但是放大图像画质劣化明显。
放大后图像的座标$(x',y')$除以放大率$a$，可以得到对应原图像的座标$(\lfloor \frac{x'}{a}\rfloor , \lfloor \frac{y'}{a}\rfloor)$,然后取整,采集这个像素值给目标图像(放大后的图)就可以得出放大图像.
使用下面的公式放大图像吧！$I'$为放大后图像，$I$为放大前图像，$a$为放大率，方括号是四舍五入取整操作：
![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%9C%80%E9%82%BB%E8%BF%91%E6%8F%92%E5%80%BC.png)

$$
I'(x,y) = I([\frac{x}{a}], [\frac{y}{a}])
$$

```Python
def nn_interpolate(img, ax=1, ay=1):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))
	y = np.round(y / ay).astype(np.int)
	x = np.round(x / ax).astype(np.int)

	out = img[y,x]

	out = out.astype(np.uint8)

	return out
```

### 26.双线性插值

双线性插值考察$4$邻域的像素点，并根据距离设置权值。虽然计算量增大使得处理时间变长，但是可以有效抑制画质劣化。

1. 放大后图像的座标$(x',y')$除以放大率$a$，可以得到对应原图像的座标$(\lfloor \frac{x'}{a}\rfloor , \lfloor \frac{y'}{a}\rfloor)$。
2. 求原图像的座标$(\lfloor \frac{x'}{a}\rfloor , \lfloor \frac{y'}{a}\rfloor)$周围$4$邻域的座标$I(x,y)$，$I(x+1,y)$，$I(x,y+1)$，$I(x+1, y+1)$：
   ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%8F%8C%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC.png)
3. 分别求这4个点与$(\frac{x'}{a}, \frac{y'}{a})$的距离，根据距离设置权重：$w = \frac{d}{\sum\ d}$
4. 根据下式求得放大后图像$(x',y')$处的像素值：

$$
d_x = \frac{x'}{a} - x\\
  d_y = \frac{y'}{a} - y\\
  I'(x',y') = (1-d_x)\  (1-d_y)\  I(x,y) + d_x\  (1-d_y)\  I(x+1,y) + (1-d_x)\  d_y\  I(x,y+1) + d_x\  d_y\  I(x+1,y+1)
$$

```Python
def bl_interpolate(img, ax=1., ay=1.):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance 
	dx = x - ix
	dy = y - iy

	dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
	dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out
```

### 27.双三次插值（ Bicubic Interpolation ）

双三次插值是双线性插值的扩展，使用邻域$16$像素进行插值。
![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%8F%8C%E4%B8%89%E6%AC%A1%E6%8F%92%E5%80%BC.png)
各自像素间的距离由下式决定：

$$
\begin{align*}
d_{x_1} = |\frac{x'}{a\  x} - (x-1)|\quad 
d_{x_2} = |\frac{x'}{a\  x}- x| \quad 
d_{x_3} = |\frac{x'}{a\  x}- (x+1)|\quad 
d_{x_4} = |\frac{x'}{a\  x} - (x+2)|\\
d_{y_1} = |\frac{x'}{a\  y} - (y-1)|\quad 
d_{y_2} = |\frac{x'}{a\  y} - y| \quad 
d_{y_3} = |\frac{x'}{a\  y} - (y+1)| \quad 
d_{y_4} = |\frac{x'}{a\  y} - (y+2)|
\end{align*}
$$

权重由基于距离的函数取得。$a$在大部分时候取$-1$。大体上说，图中蓝色像素的距离$|t|\leq 1$，绿色像素的距离$1<|t|\leq 2$：

$$
h(t)=
\begin{cases}
(a+2)\ |t|^3 - (a+3)\ |t|^2 + 1 &\text{when}\quad |t|\leq 1  \\
a\ |t|^3 - 5\  a\ |t|^2 + 8\  a\  |t| - 4\  a&\text{when}\quad 1<|t|\leq 2\\
0&\text{else}
\end{cases}
$$

利用上面得到的权重，通过下面的式子扩大图像。将每个像素与权重的乘积之和除以权重的和。

$$
I'(x', y')=\frac{1}{\sum\limits_{j=1}^4\ \sum\limits_{i=1}^4\ h(d_{xi})\ h(d_{yj})}\ \sum\limits_{j=1}^4\ \sum\limits_{i=1}^4\ I(x+i-2,y+j-2)\ h(d_{xi})\ h(d_{yj})
$$

```Python
def bc_interpolate(img, ax=1., ay=1.):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	# get positions of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))
	y = (y / ay)
	x = (x / ax)

	# get positions of original image
	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-1)
	iy = np.minimum(iy, H-1)

	# get distance of each position of original image
	dx2 = x - ix
	dy2 = y - iy
	dx1 = dx2 + 1
	dy1 = dy2 + 1
	dx3 = 1 - dx2
	dy3 = 1 - dy2
	dx4 = 1 + dx3
	dy4 = 1 + dy3

	dxs = [dx1, dx2, dx3, dx4]
	dys = [dy1, dy2, dy3, dy4]

	# bi-cubic weight
	def weight(t):
		a = -1.
		at = np.abs(t)
		w = np.zeros_like(t)
		ind = np.where(at <= 1)
		w[ind] = ((a+2) * np.power(at, 3) - (a+3) * np.power(at, 2) + 1)[ind]
		ind = np.where((at > 1) & (at <= 2))
		w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
		return w

	w_sum = np.zeros((aH, aW, C), dtype=np.float32)
	out = np.zeros((aH, aW, C), dtype=np.float32)

	# interpolate
	for j in range(-1, 3):
		for i in range(-1, 3):
			ind_x = np.minimum(np.maximum(ix + i, 0), W-1)
			ind_y = np.minimum(np.maximum(iy + j, 0), H-1)

			wx = weight(dxs[i+1])
			wy = weight(dys[j+1])
			wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
			wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

			w_sum += wx * wy
			out += wx * wy * img[ind_y, ind_x]

	out /= w_sum
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out
```

### 28.仿射变换（ Afine Transformations ）——平行移动

仿射变换利用$3\times3$的矩阵来进行图像变换。

变换的方式有平行移动（问题28）、放大缩小（问题29）、旋转（问题30）、倾斜（问题31）等。

原图像记为$(x,y)$，变换后的图像记为$(x',y')$。

图像放大缩小矩阵为下式：

$$
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)=
\left(
\begin{matrix}
a&b\\
c&d
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)
$$

另一方面，平行移动按照下面的式子计算：

$$
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)=
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)+
\left(
\begin{matrix}
t_x\\
t_y
\end{matrix}
\right)
$$

把上面两个式子盘成一个：

$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
a&b&t_x\\
c&d&t_y\\
0&0&1
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$

但是在实际操作的过程中，如果一个一个地计算原图像的像素的话，处理后的像素可能没有在原图像中有对应的坐标。[^2]

因此，我们有必要对处理后的图像中各个像素进行仿射变换逆变换，取得变换后图像中的像素在原图像中的坐标。仿射变换的逆变换如下：

$$
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)=
\frac{1}{a\  d-b\  c}\ 
\left(
\begin{matrix}
d&-b\\
-c&a
\end{matrix}
\right)\  
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)-
\left(
\begin{matrix}
t_x\\
t_y
\end{matrix}
\right)
$$

这回的平行移动操作使用下面的式子计算。$t_x$和$t_y$是像素移动的距离。

$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
1&0&t_x\\
0&1&t_y\\
0&0&1
\end{matrix}
\right)\  
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$

利用仿射变换让图像在$x$方向上$+30$，在$y$方向上$-30$吧！

```Python
def affine(img, a, b, c, d, tx, ty):
  	H, W, C = img.shape

	# temporary image
	img = np.zeros((H+2, W+2, C), dtype=np.float32)
	img[1:H+1, 1:W+1] = _img

	# get new image shape
	H_new = np.round(H * d).astype(np.int)
	W_new = np.round(W * a).astype(np.int)
	out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image
	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# get position of original image by affine
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

	x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
	y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixcel to new image
	out[y_new, x_new] = img[y, x]

	out = out[:H_new, :W_new]
	out = out.astype(np.uint8)

	return out
# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(img, a=1, b=0, c=0, d=1, tx=30, ty=-30)
```

### 29.仿射变换（ Afine Transformations ）——放大缩小

1. 使用仿射变换，将图片在$x$方向上放大$1.3$倍，在$y$方向上缩小至原来的$\frac{4}{5}$。
2. 在上面的条件下，同时在$x$方向上向右平移$30$（$+30$），在$y$方向上向上平移$30$（$-30$）。

```Python
# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(img, a=1.3, b=0, c=0, d=0.8, tx=30, ty=-30)
```

### 30.仿射变换（ Afine Transformations ）——旋转

1. 使用仿射变换，逆时针旋转$30$度。
2. 使用仿射变换，逆时针旋转$30$度并且能让全部图像显现（也就是说，单纯地做仿射变换会让图片边缘丢失，这一步中要让图像的边缘不丢失，需要耗费一些工夫）。

使用下面的式子进行逆时针方向旋转$A$度的仿射变换：

$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
\cos(A)&-\sin(A)&t_x\\
\sin(A)&\cos(A)&t_y\\
0&0&1
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# affine
def affine(img, a, b, c, d, tx, ty):
	H, W, C = _img.shape

	# temporary image
	img = np.zeros((H+2, W+2, C), dtype=np.float32)
	img[1:H+1, 1:W+1] = _img

	# get shape of new image
	H_new = np.round(H).astype(np.int)
	W_new = np.round(W).astype(np.int)
	out = np.zeros((H_new, W_new, C), dtype=np.float32)

	# get position of new image
	x_new = np.tile(np.arange(W_new), (H_new, 1))
	y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# get position of original image by affine
	adbc = a * d - b * c
	x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
	y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

	# adjust center by affine
	dcx = (x.max() + x.min()) // 2 - W // 2
	dcy = (y.max() + y.min()) // 2 - H // 2

	x -= dcx
	y -= dcy

	x = np.clip(x, 0, W + 1)
	y = np.clip(y, 0, H + 1)

	# assign pixcel
	out[y_new, x_new] = img[y, x]
	out = out.astype(np.uint8)

	return out

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)


# Affine
A = 30.
theta = - np.pi * A / 180.

out = affine(img, a=np.cos(theta), b=-np.sin(theta), c=np.sin(theta), d=np.cos(theta),
 tx=0, ty=0)


# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

```

### 31.仿射变换（Afine Transformations）——倾斜

1. 使用仿射变换，输出（1）那样的$x$轴倾斜$30$度的图像（$t_x=30$），这种变换被称为X-sharing。
2. 使用仿射变换，输出（2）那样的y轴倾斜$30$度的图像（$t_y=30$），这种变换被称为Y-sharing。
3. 使用仿射变换，输出（3）那样的$x$轴、$y$轴都倾斜$30$度的图像($t_x = 30$，$t_y = 30$)。

原图像的大小为$h\  w$，使用下面各式进行仿射变换：

* X-sharing

  $$
  a=\frac{t_x}{h}\\
  \left[
  \begin{matrix}
  x'\\
  y'\\
  1
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  1&a&t_x\\
  0&1&t_y\\
  0&0&1
  \end{matrix}
  \right]\ 
  \left[
  \begin{matrix}
  x\\
  y\\
  1
  \end{matrix}
  \right]
  $$
* Y-sharing

  $$
  a=\frac{t_y}{w}\\
  \left[
  \begin{matrix}
  x'\\
  y'\\
  1
  \end{matrix}
  \right]=\left[
  \begin{matrix}
  1&0&t_x\\
  a&1&t_y\\
  0&0&1
  \end{matrix}
  \right]\ 
  \left[
  \begin{matrix}
  x\\
  y\\
  1
  \end{matrix}
  \right]
  $$

|                                  输入                                  |                                   输出 (1)                                   |                                   输出 (2)                                   |                                   输出 (3)                                   |
| :--------------------------------------------------------------------: | :--------------------------------------------------------------------------: | :--------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_31_1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_31_2.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_31_3.jpg) |

```Python
def affine(img, dx=30, dy=30):
    # get shape
    H, W, C = img.shape

    # Affine hyper parameters
    a = 1.
    b = dx / H
    c = dy / W
    d = 1.
    tx = 0.
    ty = 0.

    # prepare temporary
    _img = np.zeros((H+2, W+2, C), dtype=np.float32)

    # insert image to center of temporary
    _img[1:H+1, 1:W+1] = img

    # prepare affine image temporary
    H_new = np.ceil(dy + H).astype(np.int)
    W_new = np.ceil(dx + W).astype(np.int)
    out = np.zeros((H_new, W_new, C), dtype=np.float32)

    # preprare assigned index
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    # prepare inverse matrix for affine
    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    # assign value from original to affine image
    out[y_new, x_new] = _img[y, x]
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(img, dx=30, dy=30)
```

### 32.傅立叶变换（Fourier Transform）

二维离散傅立叶变换是傅立叶变换在图像处理上的应用方法。通常傅立叶变换用于分离模拟信号或音频等连续一维信号的频率。但是，数字图像使用$[0,255]$范围内的离散值表示，并且图像使用$H\times W$的二维矩阵表示，所以在这里使用二维离散傅立叶变换。

二维离散傅立叶变换使用下式计算，其中$I$表示输入图像：

$$
G(k,l)=\frac{1}{H\  W}\ \sum\limits_{y=0}^{H-1}\ \sum\limits_{x=0}^{W-1}\ I(x,y)\ e^{-2\  \pi\  j\ (\frac{k\  x}{W}+\frac{l\  y}{H})}
$$

在这里让图像灰度化后，再进行离散二维傅立叶变换。

频谱图为了能表示复数$G$，所以图上所画长度为$G$的绝对值。这回的图像表示时，请将频谱图缩放至$[0,255]$范围。

二维离散傅立叶逆变换从频率分量$G$按照下式复原图像：

$$
I(x,y)=\frac{1}{H\  W}\ \sum\limits_{l=0}^{H-1}\ \sum\limits_{k=0}^{W-1}\ G(l,k)\ e^{2\  \pi\  j\ (\frac{k\  x}{W}+\frac{l\  y}{H})}
$$

上式中$\exp(j)$是个复数，实际编程的时候请务必使用下式中的绝对值形态[^1]：

如果只是简单地使用 `for`语句的话，计算量达到$128^4$，十分耗时。如果善用 `NumPy`的化，则可以减少计算量（答案中已经减少到$128^2$）。

使用离散二维傅立叶变换（Discrete Fourier Transformation），图像表示为频谱图。然后用二维离散傅立叶逆变换将图像复原。

|                                        输入                                         |                                        输出                                         |                                    频谱图                                     |
| :---------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_32_ps.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DFT hyper-parameters
K, L = 128, 128
channel = 3

# DFT 傅里叶变换
def dft(img):
	H, W, _ = img.shape

	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT 傅里叶反变换
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DFT
G = dft(img)

# write poser spectal to image 将频谱的值缩放到255的范围
ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
cv2.imwrite("out_ps.jpg", ps)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 33.傅立叶变换——低通滤波

将 `imori.jpg`灰度化之后进行傅立叶变换并进行低通滤波，之后再用傅立叶逆变换复原吧！

通过离散傅立叶变换得到的频率在左上、右上、左下、右下等地方频率较低，在中心位置频率较高。[^2]

在图像中，高频成分指的是颜色改变的地方（噪声或者轮廓等），低频成分指的是颜色不怎么改变的部分（比如落日的渐变）。在这里，使用去除高频成分，保留低频成分的**低通滤波器**吧！

在这里，假设从低频的中心到高频的距离为$r$，我们保留$0.5\ r$的低频分量。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray灰度化
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# LPF低通滤波
def lpf(G, ratio=0.5):
	H, W, _ = G.shape

	# transfer positions变换位置
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r > (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale  注：好像不进行灰度化也可以，该算法是支持三通道的
gray = bgr2gray(img)

# DFT
G = dft(img)  

# LPF
G = lpf(G)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

```

### 34.傅立叶变换——高通滤波

将 `imori.jpg`灰度化之后进行傅立叶变换并进行高通滤波，之后再用傅立叶逆变换复原吧！
与低通相比只是把中间点的颜色进行了互换
							  								![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%82%85%E9%87%8C%E5%8F%B6%E9%AB%98%E9%80%9A%E6%BB%A4%E6%B3%A2.png)
在这里，我们使用可以去除低频部分，只保留高频部分的**高通滤波器**。假设从低频的中心到高频的距离为$r$，我们保留$0.2\ r$的低频分量。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# HPF高通滤波
def hpf(G, ratio=0.1):
	H, W, _ = G.shape

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r < (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = bgr2gray(img)

# DFT
G = dft(img)

# HPF
G = hpf(G)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 35.傅立叶变换——带通滤波

![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%B8%A6%E9%80%9A%E6%BB%A4%E6%B3%A2.png)

将图片灰度化之后进行傅立叶变换并进行带通滤波，之后再用傅立叶逆变换复原吧！

在这里，我们使用可以保留介于低频成分和高频成分之间的分量的**带通滤波器**。在这里，我们使用可以去除低频部分，只保留高频部分的高通滤波器。假设从低频的中心到高频的距离为$r$，我们保留$0.1\  r$至$0.5\  r$的分量。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# BPF
def bpf(G, ratio1=0.1, ratio2=0.5):
	H, W, _ = G.shape

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[(r < (W // 2 * ratio1)) | (r > (W // 2 * ratio2))] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = bgr2gray(img)

# DFT
G = dft(img)

# BPF
G = bpf(G, ratio1=0.1, ratio2=0.5)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 36.JPEG 压缩——第一步：离散余弦变换（Discrete Cosine Transformation）

图像的离散余弦变换广泛用于图像的压缩。对原始图像进行离散余弦变换，变换后DCT系数能量主要集中在左上角，其余大部分系数接近于零，DCT具有适用于图像压缩的特性。将变换后的DCT系数进行门限操作，将小于一定值得系数归零，这就是图像压缩中的量化过程，然后进行逆DCT运算，可以得到压缩后的图像。
                                                      `<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E7%A6%BB%E6%95%A3%E4%BD%99%E5%BC%A6%E5%8F%98%E6%8D%A2.png" style="zoom:67%;" />`
图像灰度化之后，先进行离散余弦变换，再进行离散余弦逆变换吧！

离散余弦变换（Discrete Cosine Transformation）是一种使用下面式子计算的频率变换：

$$
0\leq u,\ v\leq T\\
F(u,v)=\frac{2}{T}\  C(u)\  C(v)\ \sum\limits_{y=0}^{T-1}\ \sum\limits_{x=0}^{T-1}\ I(x,y)\  \cos(\frac{(2\  x+1)\  u\  \pi}{2\  T}\ \cos(\frac{(2\  y+1)\  v\  \pi}{2\  T})\\
C(u)=
\begin{cases}
\frac{1}{\sqrt{2}}& (\text{if}\ u=0)\\
1&(\text{else})
\end{cases}
$$

离散余弦逆变换（Inverse Discrete Cosine Transformation）是离散余弦变换的逆变换，使用下式定义。

在这里，$T$是指分割的大小，$K$是决定图像复原时分辨率高低的参数。$K=T$时，DCT的系数全被保留，因此IDCT时分辨率最大。$K=1$或$K=2$时，图像复原时的信息量（DCT系数）减少，分辨率降低。如果适当地设定$K$，可以减小文件大小。

$$
1\leq K\leq T\\
f(x,y)=\frac{2}{T}\ \sum\limits_{u=0}^{K-1}\sum\limits_{v=0}^{K-1}\ C(u)\ C(v)\ F(u,v)\ \cos(\frac{(2\  x+1)\  u\  \pi}{2\  T})\ \cos(\frac{(2\  y+1)\  v\  \pi}{2\  T})\\
C(u)=
\begin{cases}
\frac{1}{\sqrt{2}}& (\text{if}\ u=0)\\
1&(\text{else})
\end{cases}
$$

在这里我们先将图像分割成$8\times8$的小块，在各个小块中使用离散余弦变换编码，使用离散余弦逆变换解码，这就是 JPEG的编码过程。现在我们也同样地，把图像分割成$8\times8$的小块，然后进行离散余弦变换和离散余弦逆变换。

> 这一整段我整体都在瞎**译，原文如下：
>
> ここでは画像を8x8ずつの領域に分割して、各領域で以上のDCT, IDCTを繰り返すことで、JPEG符号に応用される。 今回も同様に8x8の領域に分割して、DCT, IDCTを行え。
> 在此，将图像划分为8x8区域，并在每个区域中重复上述DCT和IDCT，将其应用于JPEG代码。同样，这次，您可以将DCT和IDCT分成8x8区域来执行。
> ——gzr

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 8
channel = 3

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT 离散余弦变换
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * w(x,y,u,v)

    return F


# IDCT 离散余弦逆变换
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out



# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT
F = dct(img)

# IDCT
out = idct(F)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 37.PSNR——峰值信噪比

离散余弦逆变换中如果不使用$8$作为系数，而是使用$4$作为系数的话(即K=4)，图像的画质会变差。来求输入图像和经过离散余弦逆变换之后的图像的峰值信噪比吧！再求出离散余弦逆变换的比特率吧！

峰值信噪比（Peak Signal to Noise Ratio）缩写为PSNR，用来表示信号最大可能功率和影响它的表示精度的破坏性噪声功率的比值，可以显示图像画质损失的程度。

峰值信噪比越大，表示画质损失越小。峰值信噪比通过下式定义。MAX表示图像点颜色的最大数值。如果取值范围是$[0,255]$的话，那么MAX的值就为255。MSE表示均方误差（Mean Squared Error），用来表示两个图像各个像素点之间差值平方和的平均数：

$$
\text{PSNR}=10\  \log_{10}\ \frac{{v_{max}}^2}{\text{MSE}}\\
\text{MSE}=\frac{\sum\limits_{y=0}^{H-1}\ \sum\limits_{x=0}^{W-1}\ [I_1(x,y)-I_2(x,y)]^2}{H\  W}
$$

如果我们进行$8\times8$的离散余弦变换，离散余弦逆变换的系数为$KtimesK$的话，比特率按下式定义：

$$
\text{bit rate}=8\ \frac{K^2}{8^2}
$$

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 4
channel = 3

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out


# MSE
def MSE(img1, img2):
    H, W, _ = img1.shape
    mse = np.sum((img1 - img2) ** 2) / (H * W * channel)
    return mse

# PSNR
def PSNR(mse, vmax=255):
    return 10 * np.log10(vmax * vmax / mse)

# bitrate
def BITRATE():
    return 1. * T * K * K / T / T


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT
F = dct(img)

# IDCT
out = idct(F)

# MSE
mse = MSE(img, out)

# PSNR
psnr = PSNR(mse)

# bitrate
bitrate = BITRATE()

print("MSE:", mse)
print("PSNR:", psnr)
print("bitrate:", bitrate)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 38.JPEG压缩——第二步：离散余弦变换+量化

量化离散余弦变换系数并使用 离散余弦逆变换恢复。再比较变换前后图片的大小。

量化离散余弦变换系数是用于编码 JPEG 图像的技术。

量化即在对值在预定义的区间内舍入，其中 `floor`、`ceil`、`round`等是类似的计算。

在 JPEG 图像中，根据下面所示的量化矩阵量化离散余弦变换系数。该量化矩阵取自 JPEG 软件开发联合会组织颁布的标准量化表。在量化中，将8x 8的系数除以（量化矩阵） Q 并四舍五入。之后然后再乘以 Q 。对于离散余弦逆变换，应使用所有系数。

```bash
Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
```

由于量化降低了图像的大小，因此可以看出数据量已经减少。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 4
channel = 3

# Quantization 量化
def quantization(F):
    H, W, _ = F.shape

    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                (12, 12, 14, 19, 26, 58, 60, 55),
                (14, 13, 16, 24, 40, 57, 69, 56),
                (14, 17, 22, 29, 51, 87, 80, 62),
                (18, 22, 37, 56, 68, 109, 103, 77),
                (24, 35, 55, 64, 81, 104, 113, 92),
                (49, 64, 78, 87, 103, 121, 120, 101),
                (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
  
    for ys in range(0, H, T):
        for xs in range(0, W, T):
            for c in range(channel):
                F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q
  
    return F

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT 上一个代码中有
F = dct(img)

# quantization
F = quantization(F)

# IDCT 上一个代码中有
out = idct(F)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 39.JPEG 压缩——第三步：YCbCr 色彩空间

在 YCbCr 色彩空间内，将 Y 乘以0.7以使对比度变暗。

YCbCr 色彩空间是用于将图像由表示亮度的 Y、表示蓝色色度Cb以及表示红色色度Cr表示的方法。

这用于 JPEG 转换。

使用下式从 RGB 转换到 YCbCr：

$$
Y = 0.299 \  R + 0.5870 \  G + 0.114 \  B\\
Cb = -0.1687\  R - 0.3313 \  G + 0.5 \  B + 128\\
Cr = 0.5 \  R - 0.4187 \  G - 0.0813 \  B + 128
$$

使用下式从 YCbCr 转到 RGB：

$$
R = Y + (Cr - 128) \  1.402\\
G = Y - (Cb - 128) \  0.3441 - (Cr - 128) \  0.7139\\
B = Y + (Cb - 128) \  1.7718
$$

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

channel = 3

# BGR -> Y Cb Cr
def BGR2YCbCr(img):
  H, W, _ = img.shape

  ycbcr = np.zeros([H, W, 3], dtype=np.float32)

  ycbcr[..., 0] = 0.2990 * img[..., 2] + 0.5870 * img[..., 1] + 0.1140 * img[..., 0]
  ycbcr[..., 1] = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
  ycbcr[..., 2] = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

  return ycbcr

# Y Cb Cr -> BGR
def YCbCr2BGR(ycbcr):
  H, W, _ = ycbcr.shape

  out = np.zeros([H, W, channel], dtype=np.float32)
  out[..., 2] = ycbcr[..., 0] + (ycbcr[..., 2] - 128.) * 1.4020
  out[..., 1] = ycbcr[..., 0] - (ycbcr[..., 1] - 128.) * 0.3441 - (ycbcr[..., 2] - 128.) * 0.7139
  out[..., 0] = ycbcr[..., 0] + (ycbcr[..., 1] - 128.) * 1.7718

  out = np.clip(out, 0, 255)
  out = out.astype(np.uint8)

  return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# bgr -> Y Cb Cr
ycbcr = BGR2YCbCr(img)

# process
ycbcr[..., 0] *= 0.7

# YCbCr > RGB
out = YCbCr2BGR(ycbcr)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 40.JPEG 压缩——第四步：YCbCr+离散余弦变换+量化

将图像转为 YCbCr 色彩空间之后，进行 离散余弦变换再对 Y 用 Q1 量化矩阵量化，Cb 和 Cr 用 Q2 量化矩阵量化。最后通过离散余弦逆变换对图像复原。还需比较图像的容量。算法如下：

1. 将图像从RGB色彩空间变换到YCbCr色彩空间；
2. 对YCbCr做DCT；
3. DCT之后做量化；
4. 量化之后应用IDCT；
5. IDCT之后从YCbCr色彩空间变换到RGB色彩空间。

这是实际生活中使用的减少 JPEG 数据量的方法，Q1 和 Q2 根据 JPEG 规范由以下等式定义：

```bash
Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
               (12, 12, 14, 19, 26, 58, 60, 55),
               (14, 13, 16, 24, 40, 57, 69, 56),
               (14, 17, 22, 29, 51, 87, 80, 62),
               (18, 22, 37, 56, 68, 109, 103, 77),
               (24, 35, 55, 64, 81, 104, 113, 92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)
```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 8
channel = 3


# BGR -> Y Cb Cr
def BGR2YCbCr(img):
  H, W, _ = img.shape

  ycbcr = np.zeros([H, W, 3], dtype=np.float32)

  ycbcr[..., 0] = 0.2990 * img[..., 2] + 0.5870 * img[..., 1] + 0.1140 * img[..., 0]
  ycbcr[..., 1] = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
  ycbcr[..., 2] = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

  return ycbcr

# Y Cb Cr -> BGR
def YCbCr2BGR(ycbcr):
  H, W, _ = ycbcr.shape

  out = np.zeros([H, W, channel], dtype=np.float32)
  out[..., 2] = ycbcr[..., 0] + (ycbcr[..., 2] - 128.) * 1.4020
  out[..., 1] = ycbcr[..., 0] - (ycbcr[..., 1] - 128.) * 0.3441 - (ycbcr[..., 2] - 128.) * 0.7139
  out[..., 0] = ycbcr[..., 0] + (ycbcr[..., 1] - 128.) * 1.7718

  out = np.clip(out, 0, 255)
  out = out.astype(np.uint8)

  return out


# DCT weight
def DCT_w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * DCT_w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * DCT_w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out

# Quantization
def quantization(F):
    H, W, _ = F.shape

    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                (12, 12, 14, 19, 26, 58, 60, 55),
                (14, 13, 16, 24, 40, 57, 69, 56),
                (14, 17, 22, 29, 51, 87, 80, 62),
                (18, 22, 37, 56, 68, 109, 103, 77),
                (24, 35, 55, 64, 81, 104, 113, 92),
                (49, 64, 78, 87, 103, 121, 120, 101),
                (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

    for ys in range(0, H, T):
        for xs in range(0, W, T):
            for c in range(channel):
                if(c==0):
                    F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q1) * Q1
                else:
                    F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q2) * Q2

    return F


# JPEG without Hufman coding
def JPEG(img):
    # BGR -> Y Cb Cr
    ycbcr = BGR2YCbCr(img)

    # DCT
    F = dct(ycbcr)

    # quantization
    F = quantization(F)

    # IDCT
    ycbcr = idct(F)

    # Y Cb Cr -> BGR
    out = YCbCr2BGR(ycbcr)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# JPEG
out = JPEG(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 41.Canny边缘检测：第一步————边缘强度

问题41至问题43是边缘检测方法中的一种——Canny边缘检测法的理论介绍。

1. 使用高斯滤波；
2. 在$x$方向和$y$方向上使用Sobel滤波器，在此之上求出边缘的强度和边缘的梯度；
3. 对梯度幅值进行非极大值抑制（Non-maximum suppression）来使边缘变得更细；
4. 使用滞后阈值来对阈值进行处理。

上面就是图像边缘检测的方法了。在这里我们先完成第一步和第二步。按照以下步骤进行处理：

1. 将图像进行灰度化处理；
2. 将图像进行高斯滤波（$5\times5$，$s=1.4$）；
3. 在$x$方向和$y$方向上使用Sobel滤波器，在此之上求出边缘梯度$f_x$和$f_y$。边缘梯度可以按照下式求得：
4. 使用下面的公式将梯度方向量化：

请使用 `numpy.pad()`来设置滤波器的 `padding`吧！

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny_step1(img):  #Canny边缘检测：第一步

	# Gray scale
	def BGR2GRAY(img):  #灰度化
		b = img[:, :, 0].copy()
		g = img[:, :, 1].copy()
		r = img[:, :, 2].copy()

		# Gray scale
		out = 0.2126 * r + 0.7152 * g + 0.0722 * b
		out = out.astype(np.uint8)

		return out


	# Gaussian filter for grayscale  #高斯滤波
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
		K /= (2 * np.pi * sigma * sigma)
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c]) 
		
		out = np.clip(out, 0, 255)
		out = out[pad : pad + H, pad : pad + W]
		#out = out.astype(np.uint8)

		if gray:
			out = out[..., 0]

		return out


	# sobel filter   #sobel滤波（x方向，y方向）
	def sobel_filter(img, K_size=3):
		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			#img = np.expand_dims(img, axis=-1)
			H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad : pad + H, pad : pad + W].astype(np.uint8)
		out_h = out_h[pad : pad + H, pad : pad + W].astype(np.uint8)

		return out_v, out_h


	def get_edge_angle(fx, fy):   #计算边缘梯度
		# get edge strength
		edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
		fx = np.maximum(fx, 1e-5)

		# get edge angle
		angle = np.arctan(fy / fx)

		return edge, angle


	def angle_quantization(angle):  #将梯度方向量化
		angle = angle / np.pi * 180
		angle[angle < -22.5] = 180 + angle[angle < -22.5]
		_angle = np.zeros_like(angle, dtype=np.uint8)
		_angle[np.where(angle <= 22.5)] = 0
		_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
		_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
		_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

		return _angle

	# grayscale
	gray = BGR2GRAY(img)

	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)

	# get edge strength, angle
	edge, angle = get_edge_angle(fx, fy)

	# angle quantization
	angle = angle_quantization(angle)

	return edge, angle  #一个是幅值一个是方向


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Canny (step1)
edge, angle = Canny_step1(img)

edge = edge.astype(np.uint8)
angle = angle.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", edge)
cv2.imshow("result", edge)
cv2.imwrite("out2.jpg", angle)
cv2.imshow("result2", angle)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 42.Canny边缘检测：第二步——边缘细化

在这里我们完成Canny边缘检测的第三步。
我们从在第42问中求出的边缘梯度进行非极大值抑制，来对边缘进行细化。
非极大值抑制是对除去非极大值以外的值的操作的总称（这个术语在其它的任务中也经常出现）。

在这里，我们比较我们我们所关注的地方梯度的法线方向邻接的三个像素点的梯度幅值，如果该点的梯度值不比其它两个像素大，那么这个地方的值设置为0。

也就是说，我们在注意梯度幅值$\text{edge}(x,y)$的时候，可以根据下式由梯度方向$\text{angle}(x,y)$来变换$\text{edge}(x,y)$：

* $\text{angle}(x,y)  = 0$

  如果在$\text{edge}(x,y)$、$\text{edge}(x-1,y)$、$\text{edge}(x+1,y)$中$\text{edge}(x,y)$不是最大的，那么$\text{edge}(x,y)=0$；
* $\text{angle}(x,y)  = 45$

  如果在$\text{edge}(x,y)$、$\text{edge}(x-1,y)$、$\text{edge}(x+1,y)$中$\text{edge}(x,y)$不是最大的，那么$\text{edge}(x,y)=0$；
* $\text{angle}(x,y)  = 90$

  如果在$\text{edge}(x,y)$、$\text{edge}(x-1,y)$、$\text{edge}(x+1,y)$中$\text{edge}(x,y)$不是最大的，那么$\text{edge}(x,y)=0$；
* $\text{angle}(x,y)  = 135$

  如果在$\text{edge}(x,y)$、$\text{edge}(x-1,y)$、$\text{edge}(x+1,y)$中$\text{edge}(x,y)$不是最大的，那么$\text{edge}(x,y)=0$；

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny_step2(edge,angle):

	def non_maximum_suppression(angle,edge):
		H, W = angle.shape
		_edge = edge.copy()

		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
							_edge[y, x] = 0

		return _edge

	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)

	return edge, angle


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Canny (step1)
edge, angle = Canny_step1(img)
# Canny (step2)
edge, angle = Canny_step2(edge, angle)

edge = edge.astype(np.uint8)
angle = angle.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", edge)
cv2.imshow("result", edge)
cv2.imwrite("out2.jpg", angle)
cv2.imshow("result2", angle)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 43.Canny 边缘检测：第三步——滞后阈值

在这里我们进行 Canny 边缘检测的最后一步。

在这里我们将通过设置高阈值（HT：high threshold）和低阈值（LT：low threshold）来将梯度幅值二值化。

1. 如果梯度幅值$edge(x,y)$大于高阈值的话，令$edge(x,y)=255$；
2. 如果梯度幅值$edge(x,y)$小于低阈值的话，令$edge(x,y)=0$；
3. 如果梯度幅值$edge(x,y)$介于高阈值和低阈值之间并且周围8邻域内有比高阈值高的像素点存在，令$edge(x,y)=255$；

在这里，我们使高阈值为100，低阈值为20。顺便说一句，阈值的大小需要边看结果边调整。

上面的算法就是Canny边缘检测算法了。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny(img):

	# Gray scale
	def BGR2GRAY(img):
		b = img[:, :, 0].copy()
		g = img[:, :, 1].copy()
		r = img[:, :, 2].copy()

		# Gray scale
		out = 0.2126 * r + 0.7152 * g + 0.0722 * b
		out = out.astype(np.uint8)

		return out


	# Gaussian filter for grayscale
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
		#K /= (sigma * np.sqrt(2 * np.pi))
		K /= (2 * np.pi * sigma * sigma)
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

		out = np.clip(out, 0, 255)
		out = out[pad : pad + H, pad : pad + W]
		out = out.astype(np.uint8)

		if gray:
			out = out[..., 0]

		return out


	# sobel filter
	def sobel_filter(img, K_size=3):
		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad : pad + H, pad : pad + W]
		out_v = out_v.astype(np.uint8)
		out_h = out_h[pad : pad + H, pad : pad + W]
		out_h = out_h.astype(np.uint8)

		return out_v, out_h


	def get_edge_angle(fx, fy):
		# get edge strength
		edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
		edge = np.clip(edge, 0, 255)

		fx = np.maximum(fx, 1e-10)
		#fx[np.abs(fx) <= 1e-5] = 1e-5

		# get edge angle
		angle = np.arctan(fy / fx)

		return edge, angle


	def angle_quantization(angle):
		angle = angle / np.pi * 180
		angle[angle < -22.5] = 180 + angle[angle < -22.5]
		_angle = np.zeros_like(angle, dtype=np.uint8)
		_angle[np.where(angle <= 22.5)] = 0
		_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
		_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
		_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

		return _angle


	def non_maximum_suppression(angle, edge):
		H, W = angle.shape
		_edge = edge.copy()

		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
							_edge[y, x] = 0

		return _edge

	def hysterisis(edge, HT=100, LT=30):
		H, W = edge.shape

		# Histeresis threshold
		edge[edge >= HT] = 255
		edge[edge <= LT] = 0

		_edge = np.zeros((H + 2, W + 2), dtype=np.float32)
		_edge[1 : H + 1, 1 : W + 1] = edge

		## 8 - Nearest neighbor
		nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

		for y in range(1, H+2):
				for x in range(1, W+2):
						if _edge[y, x] < LT or _edge[y, x] > HT:
								continue
						if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
								_edge[y, x] = 255
						else:
								_edge[y, x] = 0

		edge = _edge[1:H+1, 1:W+1]
					
		return edge

	# grayscale
	gray = BGR2GRAY(img)

	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)

	# get edge strength, angle
	edge, angle = get_edge_angle(fx, fy)

	# angle quantization
	angle = angle_quantization(angle)

	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)

	# hysterisis threshold
	out = hysterisis(edge, 50, 20)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Canny
edge = Canny(img)

out = edge.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

|                                  输入 (imori.jpg)                                   |                     输出 (answers_image/answer_43.jpg)                     |
| :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_43.jpg) |

### 44.霍夫变换（Hough Transform）／直线检测——第一步：霍夫变换

第44问到第46问进行霍夫直线检测算法。

霍夫变换，是将座标由直角座标系变换到极座标系，然后再根据数学表达式检测某些形状（如直线和圆）的方法。当直线上的点变换到极座标中的时候，会交于一定的$r$、$t$的点。这个点即为要检测的直线的参数。通过对这个参数进行逆变换，我们就可以求出直线方程。

> 霍夫变换进行直线检测的原理：
> ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E7%9B%B4%E7%BA%BF.png)

> 告诉你这张图里面有一条看起来挺直的线，让你指出来线在哪、线的指向是向哪个方向。你肯定可以不假思索地告诉我线的位置和方向。但是对于计算机来说这并不是一件简单的事情，因为图片在计算机中的存储形式就是【0001101111001010110101…..】，根本没有办法从这个序列中直接给出是否有直线以及直线的位置。
>
> 霍夫变换，就是一个可以让电脑自己学会找直线的算法。
> 在这样一张图中，一条线段所在的直线可以被描述成一个一次函数： y=k x + b。而找图中的直线的基本思想就是把这张图转化到一个类似(k,b)的空间中去，其中遇到一个问题就是：如果所处理的线中有竖直的线，那么就会遇到非常大的k，这个会在数值上造成一些麻烦。所以实际上是采用Hesse仿射坐标系(极坐标)，点坐标用$（ρ,θ）$

![](https://pic1.zhimg.com/80/v2-ab06b3c3e50c47115fa705cc44e33f18_720w.jpg)

> 直角坐标系中的$(x，y)$,在极坐标中变为一条曲线$r=xcosθ+ysinθ$
> 如果原图上有两个点，如图所示：

![](https://pic3.zhimg.com/80/v2-b1e19a5a8350b44ef19ea982c3efb6a6_720w.jpg)

> 那么它们对应的曲线就会经过同一点，这个点就是原图中的直线的原点距和角方向所对应的参数点，这条直线上的所有点所描出的曲线都会经过这个点。在霍夫变换后的仿射参数空间，这个点就会被点亮。我们可以通过寻找霍夫变换后的参数空间上的亮点来确定原空间上的直线的位置的方向。

<img src="https://pic4.zhimg.com/80/v2-30d581bf3b9dc78974537d41c4f2a1cb_720w.jpg" style="zoom:50%;" />
>
                                                 <img src="https://pic1.zhimg.com/80/v2-72610ce6ff332941bbe2bee33f386778_720w.jpg" style="zoom:50%;" />

> 原图上红色紫色绿色蓝色的点对应的变换曲线相交在同一个点上，这个点对应的原点距和角方向就是原图中的四个点所在的直线的方向：
> 所以一条直线的对应的极坐标图像就是下图这样的
> `<img src="https://pic3.zhimg.com/80/v2-d61e693fa68e04ea776645f39e2a78d6_720w.jpg" style="zoom: 50%;" />`
> 至此计算机就找到了直线

#### 方法如下：

1. 我们用边缘图像来对边缘像素进行霍夫变换。
2. 在霍夫变换后获取值的直方图并选择最大点。
3. 对极大点的r和t的值进行霍夫逆变换以获得检测到的直线的参数。

在这里，进行一次霍夫变换之后，可以获得直方图。算法如下：

1. 求出图像的对角线长$r_{max}$；
2. 在边缘点$(x,y)$处，$t$取遍$[0,179]$，根据下式执行霍夫变换：

   $$
   r_{ho}=x\  \cos(t)+y\  \sin(t)
   $$
3. 做一个$180\times r_{max}$大小的表，将每次按上式计算得到的表格$(t,r)$处的值加1。换句话说，这就是在进行投票。票数会在一定的地方集中。

这一次，使用 `torino.jpg`来计算投票之后的表。使用如下参数进行 Canny 边缘检测：高斯滤波器$(5\times5,s = 1.4)$，$HT = 100$，$LT = 30$。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Hough_Line_step1(edge):
	## Voting  投票
	def voting(edge):
		H, W = edge.shape
		drho = 1
		dtheta = 1

		# get rho max length    对角线长度
		rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int) 

		# hough table
		hough = np.zeros((rho_max * 2, 180), dtype=np.int)

		# get index of edge 得到边缘的位置 
		ind = np.where(edge == 255)

		## hough transformation
		for y, x in zip(ind[0], ind[1]):
				for theta in range(0, 180, dtheta):
						# get polar coordinat4s
						t = np.pi / 180 * theta   #将t转化为角度
						rho = int(x * np.cos(t) + y * np.sin(t))

						# vote
						hough[rho + rho_max, theta] += 1
				
		out = hough.astype(np.uint8)

		return out

	# voting
	out = voting(edge)

	return out

# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Canny
edge = Canny(img)#生成边缘图像

# Hough
out = Hough_Line_step1(edge)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 45.霍夫变换（Hough Transform）／直线检测——第二步：NMS

我们将在这里进行第2步。

在问题44获得的表格中，在某个地方附近集中了很多票。这里，执行提取局部最大值的操作。

这一次，提取出投票前十名的位置，并将其表示出来。

NMS 的算法如下：

1. 在该表中，如果遍历到的像素的投票数大于其8近邻的像素值，则它不变。
2. 如果遍历到的像素的投票数小于其8近邻的像素值，则设置为0。

```Python
def Hough_Line_step2(hough):
	# non maximum suppression
	def non_maximum_suppression(hough):
		rho_max, _ = hough.shape

		## non maximum suppression
		for y in range(rho_max):
			for x in range(180):
				# get 8 nearest neighbor
				x1 = max(x-1, 0)
				x2 = min(x+2, 180)
				y1 = max(y-1, 0)
				y2 = min(y+2, rho_max-1)
				if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
					pass
					#hough[y,x] = 255
				else:
					hough[y,x] = 0

		# for hough visualization
		# get top-10 x index of hough table
		ind_x = np.argsort(hough.ravel())[::-1][:20]
		# get y index
		ind_y = ind_x.copy()
		thetas = ind_x % 180
		rhos = ind_y // 180
		_hough = np.zeros_like(hough, dtype=np.int)
		_hough[rhos, thetas] = 255

		return _hough

	# non maximum suppression
	out = non_maximum_suppression(hough)

	return out


# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Canny
edge = Canny(img)

# Hough1
out = Hough_Line_step1(edge)
# Hough2
out = Hough_Line_step2(out)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 46.霍夫变换（Hough Transform）／直线检测——第三步：霍夫逆变换

这里是将问题45中得到的极大值进行霍夫逆变换之后画出得到的直线。在这里，已经通过霍夫变换检测出了直线。

算法如下：

1. 极大值点(r,t)通过下式进行逆变换：

   $$
   y = - \frac{\cos(t)}{\sin(t) } \  x +  \frac{r}{\sin(t)}\\
   x = - \frac{\sin(t)}{\cos(t) } \  y +  \frac{r}{\cos(t)}
   $$
2. 对于每个局部最大点，使$y = 0-H -1$，$x = 0-W -1$，然后执行1中的逆变换，并在输入图像中绘制检测到的直线。请将线的颜色设置为红色$(R,G,B) = (255, 0, 0)$。

   |                                                    输入 (thorino.jpg)                                                    |                        输出 (answers/answer_46.jpg)                        |
   | :----------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
   | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E5%9B%BE1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_46.jpg) |

```Python
#直线检测完整代码
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny(img):

	# Gray scale
	def BGR2GRAY(img):
		b = img[:, :, 0].copy()
		g = img[:, :, 1].copy()
		r = img[:, :, 2].copy()

		# Gray scale
		out = 0.2126 * r + 0.7152 * g + 0.0722 * b
		out = out.astype(np.uint8)

		return out


	# Gaussian filter for grayscale
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
		#K /= (sigma * np.sqrt(2 * np.pi))
		K /= (2 * np.pi * sigma * sigma)
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

		out = np.clip(out, 0, 255)
		out = out[pad : pad + H, pad : pad + W]
		out = out.astype(np.uint8)

		if gray:
			out = out[..., 0]

		return out


	# sobel filter
	def sobel_filter(img, K_size=3):
		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad : pad + H, pad : pad + W]
		out_v = out_v.astype(np.uint8)
		out_h = out_h[pad : pad + H, pad : pad + W]
		out_h = out_h.astype(np.uint8)

		return out_v, out_h


	def get_edge_angle(fx, fy):
		# get edge strength
		edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
		edge = np.clip(edge, 0, 255)

		fx = np.maximum(fx, 1e-10)
		#fx[np.abs(fx) <= 1e-5] = 1e-5

		# get edge angle
		angle = np.arctan(fy / fx)

		return edge, angle


	def angle_quantization(angle):
		angle = angle / np.pi * 180
		angle[angle < -22.5] = 180 + angle[angle < -22.5]
		_angle = np.zeros_like(angle, dtype=np.uint8)
		_angle[np.where(angle <= 22.5)] = 0
		_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
		_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
		_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

		return _angle


	def non_maximum_suppression(angle, edge):
		H, W = angle.shape
		_edge = edge.copy()

		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
							_edge[y, x] = 0

		return _edge

	def hysterisis(edge, HT=100, LT=30):
		H, W = edge.shape

		# Histeresis threshold
		edge[edge >= HT] = 255
		edge[edge <= LT] = 0

		_edge = np.zeros((H + 2, W + 2), dtype=np.float32)
		_edge[1 : H + 1, 1 : W + 1] = edge

		## 8 - Nearest neighbor
		nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

		for y in range(1, H+2):
				for x in range(1, W+2):
						if _edge[y, x] < LT or _edge[y, x] > HT:
								continue
						if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
								_edge[y, x] = 255
						else:
								_edge[y, x] = 0

		edge = _edge[1:H+1, 1:W+1]
					
		return edge

	# grayscale
	gray = BGR2GRAY(img)

	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)

	# get edge strength, angle
	edge, angle = get_edge_angle(fx, fy)

	# angle quantization
	angle = angle_quantization(angle)

	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)

	# hysterisis threshold
	out = hysterisis(edge, 100, 30)

	return out


def Hough_Line(edge, img):
	## Voting
	def voting(edge):
		H, W = edge.shape

		drho = 1
		dtheta = 1

		# get rho max length
		rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

		# hough table
		hough = np.zeros((rho_max * 2, 180), dtype=np.int)

		# get index of edge
		ind = np.where(edge == 255)

		## hough transformation
		for y, x in zip(ind[0], ind[1]):
				for theta in range(0, 180, dtheta):
						# get polar coordinat4s
						t = np.pi / 180 * theta
						rho = int(x * np.cos(t) + y * np.sin(t))

						# vote
						hough[rho + rho_max, theta] += 1
				
		out = hough.astype(np.uint8)

		return out

	# non maximum suppression
	def non_maximum_suppression(hough):
		rho_max, _ = hough.shape

		## non maximum suppression
		for y in range(rho_max):
			for x in range(180):
				# get 8 nearest neighbor
				x1 = max(x-1, 0)
				x2 = min(x+2, 180)
				y1 = max(y-1, 0)
				y2 = min(y+2, rho_max-1)
				if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
					pass
					#hough[y,x] = 255
				else:
					hough[y,x] = 0

		return hough

	def inverse_hough(hough, img):
		H, W, _ = img.shape
		rho_max, _ = hough.shape

		out = img.copy()

		# get x, y index of hough table
		ind_x = np.argsort(hough.ravel())[::-1][:20]
		ind_y = ind_x.copy()
		thetas = ind_x % 180
		rhos = ind_y // 180 - rho_max / 2

		# each theta and rho
		for theta, rho in zip(thetas, rhos):
			# theta[radian] -> angle[degree]
			t = np.pi / 180. * theta

			# hough -> (x,y)
			for x in range(W):
				if np.sin(t) != 0:
					y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
					y = int(y)
					if y >= H or y < 0:
						continue
					out[y, x] = [0, 0, 255]
			for y in range(H):
				if np.cos(t) != 0:
					x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
					x = int(x)
					if x >= W or x < 0:
						continue
					out[y, x] = [0, 0, 255]
	
		out = out.astype(np.uint8)

		return out


	# voting
	hough = voting(edge)

	# non maximum suppression
	hough = non_maximum_suppression(hough)

	# inverse hough
	out = inverse_hough(hough, img)

	return out


# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Canny
edge = Canny(img)

# Hough
out = Hough_Line(edge, img)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 47.膨胀（Dilate）

在形态学处理的过程中，二值化图像中白色（255）的部分向$4-$近邻（上下左右）膨胀或收缩一格 。

反复进行膨胀和收缩操作，可以消除独立存在的白色像素点（见问题四十九：开操作）；或者连接白色像素点（见问题五十：闭操作）。

形态学处理中的膨胀算法如下。对于待操作的像素$I(x,y)=0$，$I(x, y-1)$，$I(x-1, y)$，$ I(x+1, y)$，$I(x, y+1)$中不论哪一个为$255$，令$I(x,y)=255$。

<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%86%A8%E8%83%80.png" style="zoom:67%;" />

换句话说，如果将上面的操作执行两次，则可以扩大两格。

在实际进行形态学处理的时候，待操作的像素$4-$近邻与矩阵$\left[\begin{matrix}0&1&0\\1&0&1\\0&1&0\end{matrix}\right]$相乘，结果大于$255$的话，将中心像素设为$255$。

将 `imori.jpg`大津二值化之后，进行两次形态学膨胀处理。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gray scale 灰度化
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# Otsu Binalization 大津二值化
def otsu_binarization(img, th=128):
	H, W = img.shape
	out = img.copy()

	max_sigma = 0
	max_t = 0

	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (H * W)
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255

	return out


# Morphology Erode 膨胀处理
def Morphology_Erode(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
					out[y-1, x-1] = 255

	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)


# Grayscale
gray = BGR2GRAY(img)

# Otsu's binarization
otsu = otsu_binarization(gray)

# Morphology - dilate
out = Morphology_Erode(otsu, Dil_time=2)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 48.腐蚀（Erode）

形态学处理中腐蚀操作如下：对于待操作的像素$I(x,y)=255$，$I(x, y-1)$，$I(x-1, y)$，$ I(x+1, y)$，$I(x, y+1)$中不论哪一个不为$255$，令$I(x,y)=0$。

<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%85%90%E8%9A%80.png" style="zoom:50%;" />

在实际进行形态学处理的时候，待操作的像素$4-$近邻与矩阵$\left[\begin{matrix}0&1&0\\1&0&1\\0&1&0\end{matrix}\right]$相乘，结果小于$255\times 4$的话，将中心像素设为$0$。

将 `imori.jpg`大津二值化之后，进行两次形态学腐蚀处理。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gray scale
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# Otsu Binalization
def otsu_binarization(img, th=128):
	H, W = img.shape
	out = img.copy()

	max_sigma = 0
	max_t = 0

	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (H * W)
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255

	return out


# Morphology Dilate
def Morphology_Dilate(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
					out[y-1, x-1] = 0

	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Grayscale
gray = BGR2GRAY(img)

# Otsu's binarization
otsu = otsu_binarization(gray)

# Morphology - dilate
out = Morphology_Dilate(otsu, Erode_time=2)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 49.开运算（Opening Operation）

大津二值化之后，进行开运算（N=1）吧。

开运算，即先进行N次腐蚀再进行N次膨胀。

![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%BC%80%E8%BF%90%E7%AE%97.png)

开运算可以用来去除仅存的小块像素。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gray scale
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# Otsu Binalization
def otsu_binarization(img, th=128):
	H, W = img.shape
	out = img.copy()

	max_sigma = 0
	max_t = 0

	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (H * W)
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255

	return out


# Morphology Dilate
def Morphology_Dilate(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
					out[y-1, x-1] = 0

	return out


# Morphology Erode
def Morphology_Erode(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
					out[y-1, x-1] = 255

	return out


# Opening morphology
def Morphology_Opening(img, time=1):
	out = Morphology_Dilate(img, Erode_time=time)
	out = Morphology_Erode(out, Dil_time=time)
	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Grayscale
gray = BGR2GRAY(img)

# Otsu's binarization
otsu = otsu_binarization(gray)

# Morphology - opening
out = Morphology_Opening(otsu, time=1)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 50.闭运算（Closing Operation）

Canny边缘检测之后，进行$N=1$的闭处理吧。

闭运算，即先进行$N$次膨胀再进行$N$次腐蚀。

<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E9%97%AD%E8%BF%90%E7%AE%97.png" style="zoom:67%;" />

闭运算能够将中断的像素连接起来。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Canny
def Canny(img):

	# Gray scale
	def BGR2GRAY(img):
		b = img[:, :, 0].copy()
		g = img[:, :, 1].copy()
		r = img[:, :, 2].copy()

		# Gray scale
		out = 0.2126 * r + 0.7152 * g + 0.0722 * b
		out = out.astype(np.uint8)

		return out


	# Gaussian filter for grayscale
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
			gray = False
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape
			gray = True

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
		#K /= (sigma * np.sqrt(2 * np.pi))
		K /= (2 * np.pi * sigma * sigma)
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

		out = np.clip(out, 0, 255)
		out = out[pad : pad + H, pad : pad + W]
		out = out.astype(np.uint8)

		if gray:
			out = out[..., 0]

		return out


	# sobel filter
	def sobel_filter(img, K_size=3):
		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad : pad + H, pad : pad + W]
		out_v = out_v.astype(np.uint8)
		out_h = out_h[pad : pad + H, pad : pad + W]
		out_h = out_h.astype(np.uint8)

		return out_v, out_h


	def get_edge_angle(fx, fy):
		# get edge strength
		edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
		edge = np.clip(edge, 0, 255)

		fx = np.maximum(fx, 1e-10)
		#fx[np.abs(fx) <= 1e-5] = 1e-5

		# get edge angle
		angle = np.arctan(fy / fx)

		return edge, angle


	def angle_quantization(angle):
		angle = angle / np.pi * 180
		angle[angle < -22.5] = 180 + angle[angle < -22.5]
		_angle = np.zeros_like(angle, dtype=np.uint8)
		_angle[np.where(angle <= 22.5)] = 0
		_angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
		_angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
		_angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

		return _angle


	def non_maximum_suppression(angle, edge):
		H, W = angle.shape
		_edge = edge.copy()

		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
							_edge[y, x] = 0

		return _edge

	def hysterisis(edge, HT=100, LT=30):
		H, W = edge.shape

		# Histeresis threshold
		edge[edge >= HT] = 255
		edge[edge <= LT] = 0

		_edge = np.zeros((H + 2, W + 2), dtype=np.float32)
		_edge[1 : H + 1, 1 : W + 1] = edge

		## 8 - Nearest neighbor
		nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

		for y in range(1, H+2):
				for x in range(1, W+2):
						if _edge[y, x] < LT or _edge[y, x] > HT:
								continue
						if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
								_edge[y, x] = 255
						else:
								_edge[y, x] = 0

		edge = _edge[1:H+1, 1:W+1]
					
		return edge

	# grayscale
	gray = BGR2GRAY(img)

	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)

	# get edge strength, angle
	edge, angle = get_edge_angle(fx, fy)

	# angle quantization
	angle = angle_quantization(angle)

	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)

	# hysterisis threshold
	out = hysterisis(edge, 50, 20)

	return out


# Morphology Dilate
def Morphology_Dilate(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
					out[y-1, x-1] = 0

	return out


# Morphology Erode
def Morphology_Erode(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H+1):
			for x in range(1, W+1):
				if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
					out[y-1, x-1] = 255

	return out

# Morphology Closing
def Morphology_Closing(img, time=1):
	out = Morphology_Erode(img, Dil_time=time)
	out = Morphology_Dilate(out, Erode_time=time)
	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Canny
canny = Canny(img)

# Morphology - opening
out = Morphology_Closing(canny, time=1)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 51.形态学梯度（Morphology Gradient）

在进行大津二值化之后，计算图像的形态学梯度吧。

形态学梯度为经过膨胀操作（dilate）的图像与经过腐蚀操作（erode）的图像的差，可以用于抽出物体的边缘。

在这里，形态学处理的核 `N=1`。

|                                        输入                                         |                                    输出                                    |
| :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_51.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Otsu binary 大津二值化
## Grayscale 灰度化
out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

## Determine threshold of Otsu's binarization
max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = out[np.where(out < _t)[0]]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H * W)
    v1 = out[np.where(out >= _t)[0]]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H * W)
    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

## Binarization 二值化
#print("threshold >>", max_t)
th = max_t
out[out < th] = 0
out[out >= th] = 255

# Morphology filter  模板
MF = np.array(((0, 1, 0),
               (1, 0, 1),
               (0, 1, 0)), dtype=np.int)

# Morphology - erode  腐蚀操作
Erode_time = 1
erode = out.copy()

for i in range(Erode_time):
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                erode[y-1, x-1] = 0
          
# Morphology - dilate  膨胀操作
Dil_time = 1
dilate = out.copy()

for i in range(Dil_time):
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                dilate[y-1, x-1] = 255
          
out = np.abs(erode - dilate) * 255
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 52.顶帽（Top Hat）

在进行大津二值化之后，进行顶帽运算吧。

顶帽运算是原图像与开运算的结果图的差。

在这里，我们求大津二值化之后的图像和开处理（`N=3`）之后的图像的差，可以提取出细线状的部分或者噪声。

 样例图片不好突出显示顶帽运算的效果，如果找到了其它适合的图像会在这里作出更正。

|                                  输入 (imori.jpg)                                   |                        输出(answers/answer_52.jpg)                         |
| :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_52.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Otsu binary
## Grayscale
out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

## Determine threshold of Otsu's binarization
max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = out[np.where(out < _t)]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H * W)
    v1 = out[np.where(out >= _t)]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H * W)
    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

## Binarization
#print("threshold >>", max_t)
th = max_t
out[out < th] = 0
out[out >= th] = 255

# Morphology filter
MF = np.array(((0, 1, 0),
               (1, 0, 1),
               (0, 1, 0)), dtype=np.int)

# Morphology - erode
Erode_time = 3
mor = out.copy()

for i in range(Erode_time):
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                mor[y-1, x-1] = 0

# Morphology - dilate
Dil_time = 3

for i in range(Dil_time):
    tmp = np.pad(mor, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                mor[y-1, x-1] = 255

out = out - mor
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 53.黑帽（Black Hat）

在进行大津二值化之后，进行黑帽运算吧。

黑帽运算是原图像与闭运算的结果图的差。

在这里，我们求大津二值化之后的图像和闭处理（`N=3`）之后的图像的差，在这里和顶帽运算一样，可以提取出细线状的部分或者噪声。

|                                  输入 (imori.jpg)                                   |                        输出(answers/answer_52.jpg)                         |
| :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_53.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Otsu binary
## Grayscale
out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

## Determine threshold of Otsu's binarization
max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = out[np.where(out < _t)]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H * W)
    v1 = out[np.where(out >= _t)]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H * W)
    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

## Binarization
#print("threshold >>", max_t)
th = max_t
out[out < th] = 0
out[out >= th] = 255

# Morphology filter
MF = np.array(((0, 1, 0),
               (1, 0, 1),
               (0, 1, 0)), dtype=np.int)

# Morphology - dilate
Dil_time = 3
mor = out.copy()

for i in range(Dil_time):
    tmp = np.pad(out, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                mor[y-1, x-1] = 255

# Morphology - erode
Erode_time = 3

for i in range(Erode_time):
    tmp = np.pad(mor, (1, 1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):
            if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                mor[y-1, x-1] = 0

out = mor - out
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 54.使用误差平方和算法（Sum of Squared Difference）进行模式匹配（Template Matching）

模式匹配，即寻找待匹配图像和全体图像中最相似的部分，用于物体检测任务。现在虽然使用卷积神经网络（`CNN`）来检测物体，但是模式识别仍然是最基本的处理方法。

下面介绍具体算法。原图像记为$I(H\times W)$，待匹配图像为$T(h\times w)$：

1. 对于图像$I$：，`for ( j = 0, H-h)  for ( i = 0, W-w)`在一次移动1像素的过程中，原图像I的一部分$I(i:i+w, j:j+h)$与待匹配图像计算相似度$S$。
2. S最大或最小的地方即为匹配的位置。

S的计算方法主要有 `SSD`、`SAD`（第55题）、`NCC`（第56题）、`ZNCC`（第57题）等。对于不同的方法，我们需要选择出最大值或者最小值。

在这里我们使用误差平方和 `SSD`（Sum of Squared Difference）。`SSD`计算像素值的差的平方和，S取误差平方和最小的地方。

$$
S=\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ [I(i+x,j+y)-T(x,y)]^2
$$

顺便说一句，像模式匹配这样，从图像的左上角开始往右进行顺序查找的操作一般称作光栅扫描（Raster Scan）或者滑动窗口扫描。这样的术语在图像处理邻域经常出现。

可以使用 `cv2.rectangle ()`来画矩形。另外，`imoripart.jpg`稍微改变了颜色。

在这里我们使用误差平方和进行模式匹配。将 `imoripart.jpg`在 `imori.jpg`中匹配的图像使用红框框出来。

|                                  输入 (imori.jpg)                                   |                        template图像(imori_part.jpg)                         |                        输出(answers/answer_54.jpg)                         |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_part.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_55.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image 读入图像
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Read templete image 读入要匹配的小图像
temp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = temp.shape


# Templete matching
i, j = -1, -1
v = 255 * H * W * C #v用来记录最小值
for y in range(H-Ht):  #窗口滑动
    for x in range(W-Wt):
        _v = np.sum((img[y:y+Ht, x:x+Wt] - temp) ** 2)
        if _v < v:
            v = _v
            i, j = x, y

out = img.copy()
cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=1)##画框
out = out.astype(np.uint8)
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 55.使用绝对值差和（Sum of Absolute Differences）进行模式匹配

绝对值差和（Sum of Absolute Differences）计算像素值差的绝对值之和，选取$S$**最小**的位置作为匹配。

$$
S=\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ |I(i+x,j+y)-T(x,y)|
$$

在这里我们使用绝对值差和进行模式匹配。将 `imoripart.jpg`在 `imori.jpg`中匹配的图像使用红框框出来。

|                                  输入 (imori.jpg)                                   |                        template图像(imori_part.jpg)                         |                        输出(answers/answer_54.jpg)                         |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_part.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_55.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Read templete image
temp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = temp.shape

# Templete matching
i, j = -1, -1
v = 255 * H * W * C
for y in range(H-Ht):
    for x in range(W-Wt):
        _v = np.sum(np.abs(img[y:y+Ht, x:x+Wt] - temp))
        if _v < v:
            v = _v
            i, j = x, y

out = img.copy()
cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=1)
out = out.astype(np.uint8)
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

###56.使用归一化交叉相关（Normalization Cross Correlation）进行模式匹配

归一化交叉相关（Normalization Cross Correlation）求出两个图像的相似度，匹配S**最大**处的图像：

$$
S=\frac{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ |I(i+x,j+y)\  T(x,y)|}{\sqrt{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ I(i+x,j+y)^2}\  \sqrt{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ T(i,j)^2}}
$$

$S$最后的范围在$-1\leq S<=1$。`NCC`对变化十分敏感。

在这里我们使用归一化交叉相关进行模式匹配。将 `imoripart.jpg`在 `imori.jpg`中匹配的图像使用红框框出来。

|                                  输入 (imori.jpg)                                   |                        template图像(imori_part.jpg)                         |                        输出(answers/answer_54.jpg)                         |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_part.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_55.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Read templete image
temp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = temp.shape


# Templete matching
i, j = -1, -1
v = -1
for y in range(H-Ht):
    for x in range(W-Wt):
        _v = np.sum(img[y:y+Ht, x:x+Wt] * temp)
        _v /= (np.sqrt(np.sum(img[y:y+Ht, x:x+Wt]**2)) * np.sqrt(np.sum(temp**2)))
        if _v > v:
            v = _v
            i, j = x, y

out = img.copy()
cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=1)
out = out.astype(np.uint8)
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 57.使用零均值归一化交叉相关（Zero-mean Normalization Cross Correlation）进行模式匹配

在这里我们使用零均值归一化交叉相关进行模式匹配。将 `imoripart.jpg`在 `imori.jpg`中匹配的图像使用红框框出来。

零均值归一化交叉相关（Zero-mean Normalization Cross Correlation）求出两个图像的相似度，匹配$S$最大处的图像。

图像$I$的平均值记为$m_i$，图像$T$的平均值记为$m_t$。使用下式计算$S$：

$$
S=\frac{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ |[I(i+x,j+y)-m_i]\  [T(x,y)-m_t]}{\sqrt{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ [I(i+x,j+y)-m_i]^2}\  \sqrt{\sum\limits_{x=0}^w\ \sum\limits_{y=0}^h\ [T(x,y)-m_t]^2}}
$$

S最后的范围在$-1\leq S\leq 1$。零均值归一化积相关去掉平均值的话就是归一化交叉相关，据说这比归一化交叉相关对变换更加敏感。

|                                  输入 (imori.jpg)                                   |                        template图像(imori_part.jpg)                         |                        输出(answers/answer_54.jpg)                         |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_part.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_57.jpg) |

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

mi = np.mean(img)

# Read templete image
temp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = temp.shape

mt = np.mean(temp)

# Templete matching
i, j = -1, -1
v = -1
for y in range(H-Ht):
    for x in range(W-Wt):
        _v = np.sum((img[y:y+Ht, x:x+Wt]-mi) * (temp-mt))
        _v /= (np.sqrt(np.sum((img[y:y+Ht, x:x+Wt]-mi)**2)) * np.sqrt(np.sum((temp-mt)**2)))
        if _v > v:
            v = _v
            i, j = x, y

out = img.copy()
cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=1)
out = out.astype(np.uint8)
          
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

### 58.$4-$邻域连通域标记

连通域标记（Connected Component Labeling）是将邻接的像素打上相同的标记的作业。

也就是说：

```bash
黒　黒　黒　黒
黒　白　白　黒
黒　白　黒　黒
黒　黒　黒　黒
```

将相邻的白色像素打上相同的标记。

像这样的像素组成的被标记的块被称为连通区域（Connected Component）。

在这里我们为4邻域的像素打上标记。另，在这里我们使用一种被称为Lookup Table的东西。

Lookup Table是这样的：

| Source | Distination |
| :----: | :---------: |
|   1    |      1      |
|   2    |      2      |
|   3    |      1      |

一开始被打上1标签的像素（即 `Source=1`的像素）最终被分配到的标签为1（`Distination=1`）；一开始被打上3标签的像素（即 `Source =3`的像素）最终被分配的的标签也为1（`Distination=1`）。

算法如下：

1. 从左上角开始进行光栅扫描。
2. 如果当前遍历到的像素 `i(x,y)`是黑像素的什么也不干。如果是白像素，考察该像素的上方像素 `i(x,y-1)`和左边像素 `i(x-1,y)`，如果两个的取值都为0，将该像素分配一个新的标签。

   > 在这里我们用数字做标签，即1,2。原文是说“最後に割り当てたラベル + 1 を割り当てる”，直译就是分配给该像素将最后分配的标签加1数值的标签。
   >
   > ——gzr
   >
3. 如果两个像素中有一个不为0（也就是说已经分配了标签），将上方和左边的像素分配的标签中数值较小的那一个（0除外）分配给当前遍历到的像素 `i(x,y)`。在这里，将上方像素和左边像素的标签写入 `Lookup Table`的 `Source`，将当前遍历的像素 `i(x,y)`分配的标签写入 `Distination`。
4. 最后，对照 `Lookup Table`，对像素分配的标签由 `Source`变为 `Distination`。

像这样的话，邻接像素就可以打上同样的标签了。因为这里是做$4-$邻域连通域标记，所以我们只用考察上方像素和左边像素。

将 `seg.png`进行$4-$邻域连通域标记吧。

|                            输入 (seg.png)                            |                     输出(answers/answer_58.png)                     |
| :------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/seg.png) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_58.png) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("seg.png").astype(np.float32)
H, W, C = img.shape

label = np.zeros((H, W), dtype=np.int)
label[img[..., 0]>0] = 1

LUT = [0 for _ in range(H*W)]

n = 1

for y in range(H):
    for x in range(W):
        if label[y, x] == 0:
            continue
        c3 = label[max(y-1,0), x]
        c5 = label[y, max(x-1,0)]
        if c3 < 2 and c5 < 2:
            n += 1
            label[y, x] = n
        else:
            _vs = [c3, c5]
            vs = [a for a in _vs if a > 1]
            v = min(vs)
            label[y, x] = v
      
            minv = v
            for _v in vs:
                if LUT[_v] != 0:
                    minv = min(minv, LUT[_v])
            for _v in vs:
                LUT[_v] = minv
          
count = 1

for l in range(2, n+1):
    flag = True
    for i in range(n+1):
        if LUT[i] == l:
            if flag:
                count += 1
                flag = False
            LUT[i] = count

COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
out = np.zeros((H, W, C), dtype=np.uint8)

for i, lut in enumerate(LUT[2:]):
    out[label == (i+2)] = COLORS[lut-2]
  
# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 59.$8-$邻域连通域标记

在这里我们将问题58变为$8-$邻域连通域标记。

要进行$8-$邻域连通域标记，我们需要考察 `i(x-1,y-1)`，`i(x, y-1)`，`i(x+1,y-1)`，`i(x-1,y)`这4个像素。

|                                 输入                                 |                                    输出                                    |
| :------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/seg.png) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_59.png) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("seg.png").astype(np.float32)
H, W, C = img.shape

label = np.zeros((H, W), dtype=np.int)
label[img[..., 0]>0] = 1

LUT = [0 for _ in range(H*W)]

n = 1

for y in range(H):
    for x in range(W):
        if label[y, x] == 0:
            continue
        c2 = label[max(y-1,0), min(x+1, W-1)]
        c3 = label[max(y-1,0), x]
        c4 = label[max(y-1,0), max(x-1,0)]
        c5 = label[y, max(x-1,0)]
        if c3 < 2 and c5 < 2 and c2 < 2 and c4 < 2:
            n += 1
            label[y, x] = n
        else:
            _vs = [c3, c5, c2, c4]
            vs = [a for a in _vs if a > 1]
            v = min(vs)
            label[y, x] = v

            minv = v
            for _v in vs:
                if LUT[_v] != 0:
                    minv = min(minv, LUT[_v])
            for _v in vs:
                LUT[_v] = minv
          
count = 1

for l in range(2, n+1):
    flag = True
    for i in range(n+1):
        if LUT[i] == l:
            if flag:
                count += 1
                flag = False
            LUT[i] = count

COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
out = np.zeros((H, W, C), dtype=np.uint8)

for i, lut in enumerate(LUT[2:]):
    out[label == (i+2)] = COLORS[lut-2]
  
# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

### 60.透明混合（Alpha Blending）

透明混合即通过设定透明度（Alpha值）来设定图像透明度的方法。在OpenCV中虽然没有透明度这个参数，但在PIL等库中有。在这里我们手动设定透明度。

将两张图片重合的时候，这个方法是有效的。

将 `img1`和 `img2`按1:1的比例重合的时候，使用下面的式子。通过改变 Alpha 值，你可以更改两张图片重叠的权重。

```bash
alpha = 0.5
out = img1 * alpha + img2 * (1 - alpha)
```

将 `imori.jpg`和 `thorino.jpg`按照$6:4$的比例透明混合吧。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

img2 = cv2.imread("thorino.jpg").astype(np.float32)

a = 0.6
out = img * a + img2 * (1 - a)
out = out.astype(np.uint8)
  
```

### 61.$4-$连接数[^0]

$4-$连接数可以用于显示附近像素的状态。通常，对于中心像素$x_0(x，y)$不为零的情况，邻域定义如下：

$$
\begin{matrix}
x_4(x-1,y-1)& x_3(x,y-1)& x_2(x+1,y-1)\\
x_5(x-1,y) &  x_0(x,y)  & x_1(x+1,y)\\
x_6(x-1,y+1)& x_7(x,y+1)& x_8(x+1,y+1)
\end{matrix}
$$

这里，$4-$连接数通过以下等式计算：

$$
S = (x_1 - x_1\ x_2\  x_3) + (x_3 - x_3\  x_4\  x_5) + (x_5 - x_5\  x_6\  x_7) + (x_7 - x_7\  x_8\  x_1)
$$

$S$的取值范围为$[0,4]$：

- $S = 0$： 内部点；
- $S = 1$：端点；
- $S = 2$：连接点；
- $S = 3$：分支点；
- $S = 4$：交叉点。

请根据$4-$连接数将 `renketsu.png`上色。

|                                       输入 (renketsu.png)                                        |                                    输出(answers/answer_61.png)                                    |
| :----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| `<img src="https://img.geek-docs.com/opencv/opencv-examples/renketsu.png" style="zoom:600%;" />` | `<img src="https://img.geek-docs.com/opencv/opencv-examples/answer_61.png" style="zoom:600%;" />` |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Connect 4
def connect_4(img):
    # get shape
    H, W, C = img.shape

    # prepare temporary image
    tmp = np.zeros((H, W), dtype=np.int)

    # binarize
    tmp[img[..., 0] > 0] = 1

    # prepare out image
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # each pixel
    for y in range(H):
        for x in range(W):
            if tmp[y, x] < 1:
                continue

            S = 0
            S += (tmp[y,min(x+1,W-1)] - tmp[y,min(x+1,W-1)] * tmp[max(y-1,0),min(x+1,W-1)] * tmp[max(y-1,0),x])
            S += (tmp[max(y-1,0),x] - tmp[max(y-1,0),x] * tmp[max(y-1,0),max(x-1,0)] * tmp[y,max(x-1,0)])
            S += (tmp[y,max(x-1,0)] - tmp[y,max(x-1,0)] * tmp[min(y+1,H-1),max(x-1,0)] * tmp[min(y+1,H-1),x])
            S += (tmp[min(y+1,H-1),x] - tmp[min(y+1,H-1),x] * tmp[min(y+1,H-1),min(x+1,W-1)] * tmp[y,min(x+1,W-1)])
      
            if S == 0:
                out[y,x] = [0, 0, 255]
            elif S == 1:
                out[y,x] = [0, 255, 0]
            elif S == 2:
                out[y,x] = [255, 0, 0]
            elif S == 3:
                out[y,x] = [255, 255, 0]
            elif S == 4:
                out[y,x] = [255, 0, 255]
              
    out = out.astype(np.uint8)

    return out



# Read image
img = cv2.imread("renketsu.png").astype(np.float32)

# connect 4
out = connect_4(img)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 62.$8-$连接数

这里，$8-$连接数通过以下等式，将各个$x_*$的值反转$0$和$1$计算：

$$
S = (x_1 - x_1\ x_2\ x_3) + (x_3 - x_3\ x_4\ x_5) + (x_5 - x_5\ x_6\ x_7) + (x_7 - x_7\ x_8\ x_1)
$$

请根据$8-$连接数将 `renketsu.png`上色。

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# connect 8
def connect_8(img):
    # get shape
    H, W, C = img.shape

    # prepare temporary
    _tmp = np.zeros((H, W), dtype=np.int)

    # get binarize
    _tmp[img[..., 0] > 0] = 1

    # inverse for connect 8
    tmp = 1 - _tmp

    # prepare image
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # each pixel
    for y in range(H):
        for x in range(W):
            if _tmp[y, x] < 1:
                continue

            S = 0
            S += (tmp[y,min(x+1,W-1)] - tmp[y,min(x+1,W-1)] * tmp[max(y-1,0),min(x+1,W-1)] * tmp[max(y-1,0),x])
            S += (tmp[max(y-1,0),x] - tmp[max(y-1,0),x] * tmp[max(y-1,0),max(x-1,0)] * tmp[y,max(x-1,0)])
            S += (tmp[y,max(x-1,0)] - tmp[y,max(x-1,0)] * tmp[min(y+1,H-1),max(x-1,0)] * tmp[min(y+1,H-1),x])
            S += (tmp[min(y+1,H-1),x] - tmp[min(y+1,H-1),x] * tmp[min(y+1,H-1),min(x+1,W-1)] * tmp[y,min(x+1,W-1)])
      
            if S == 0:
                out[y,x] = [0, 0, 255]
            elif S == 1:
                out[y,x] = [0, 255, 0]
            elif S == 2:
                out[y,x] = [255, 0, 0]
            elif S == 3:
                out[y,x] = [255, 255, 0]
            elif S == 4:
                out[y,x] = [255, 0, 255]
              
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("renketsu.png").astype(np.float32)

# connect 8
out = connect_8(img)


# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 63.细化处理

细化是将线条宽度设置为1的过程，按照下面的算法进行处理：

1. 从左上角开始进行光栅扫描；
2. 如果$x_0(x,y)=0$，不处理。如果$x_0(x,y)=1$，满足下面三个条件时，令$x_0=0$：
   * $4-$近邻像素的取值有一个以上为$0$；
   * $x_0$的$4-$连接数为$1$；
   * x0的$8-$近邻中有三个以上取值为$1$。
3. 重复光栅扫描，直到步骤2中像素值改变次数为$0$。

用于细化的算法有Hilditch算法（问题六十四），Zhang-Suen算法（问题六十五），田村算法等。

将 `gazo.png`进行细化处理吧！

|                        输入 (gazo.png)                         |                     输出(answers/answer_63.png)                     |
| :------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](https://img.geek-docs.com/opencv/opencv-examples/gazo.png) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_63.png) |

```Python
# thining algorythm
def thining(img):
    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    count = 1
    while count > 0:
        count = 0
        tmp = out.copy()
        # each pixel ( rasta scan )
        for y in range(H):
            for x in range(W):
                # skip black pixel
                if out[y, x] < 1:
                    continue
          
                # count satisfied conditions
                judge = 0
          
                ## condition 1  计算4-近邻
                if (tmp[y, min(x+1, W-1)] + tmp[max(y-1, 0), x] + tmp[y, max(x-1, 0)] + tmp[min(y+1, H-1), x]) < 4:
                    judge += 1
              
                ## condition 2   计算4-连接数
                c = 0
                c += (tmp[y,min(x+1, W-1)] - tmp[y, min(x+1, W-1)] * tmp[max(y-1, 0),min(x+1, W-1)] * tmp[max(y-1, 0), x])
                c += (tmp[max(y-1,0), x] - tmp[max(y-1,0), x] * tmp[max(y-1, 0), max(x-1, 0)] * tmp[y, max(x-1, 0)])
                c += (tmp[y, max(x-1, 0)] - tmp[y,max(x-1, 0)] * tmp[min(y+1, H-1), max(x-1, 0)] * tmp[min(y+1, H-1), x])
                c += (tmp[min(y+1, H-1), x] - tmp[min(y+1, H-1), x] * tmp[min(y+1, H-1), min(x+1, W-1)] * tmp[y, min(x+1, W-1)])
                if c == 1:
                    judge += 1
              
                ##x condition 3   计算8-近邻
                if np.sum(tmp[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 4:
                    judge += 1
          
                # if all conditions are satisfied
                if judge == 3:  #统计三个条件是否都满足
                    out[y, x] = 0
                    count += 1

    out = out.astype(np.uint8) * 255

    return out


# Read image
img = cv2.imread("gazo.png").astype(np.float32)

# thining
out = thining(img)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 64.Hilditch 细化算法

1. 从左上角开始进行光栅扫描；
2. $x_0(x,y)=0$的话、不进行处理。$x_0(x,y)=1$的话，下面五个条件都满足的时候令$x_0=-1$：
   * 当前像素的$4-$近邻中有一个以上$0$；
   * $x_0$的$8-$连接数为$1$；
   * $x_1$至$x_8$の绝对值之和大于$2$；

* $8-$近邻像素的取值有一个以上为$1$；
  * 对所有$x_n$($n\in [1,8]$)以下任一项成立：

    * $x_n$不是$-1$；
    * 当$x_n$为$0$时，$x_0$的$8-$连接数为$1$。

3. 将每个像素的$-1$更改为$0$；
4. 重复进行光栅扫描，直到某一次光栅扫描中步骤3的变化数变为$0$。
5. 

将 `gazo.png`进行Hilditch细化算法处理吧！算法如下：

```Python
# hilditch thining
def hilditch(img):
    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse pixel value
    tmp = out.copy()
    _tmp = 1 - tmp

    count = 1
    while count > 0:
        count = 0
        tmp = out.copy()
        _tmp = 1 - tmp

        tmp2 = out.copy()
        _tmp2 = 1 - tmp2
  
        # each pixel
        for y in range(H):
            for x in range(W):
                # skip black pixel
                if out[y, x] < 1:
                    continue
          
                judge = 0
          
                ## condition 1
                if (tmp[y, min(x+1, W-1)] * tmp[max(y-1,0 ), x] * tmp[y, max(x-1, 0)] * tmp[min(y+1, H-1), x]) == 0:
                    judge += 1
              
                ## condition 2
                c = 0
                c += (_tmp[y, min(x+1, W-1)] - _tmp[y, min(x+1, W-1)] * _tmp[max(y-1, 0), min(x+1, W-1)] * _tmp[max(y-1, 0), x])
                c += (_tmp[max(y-1, 0), x] - _tmp[max(y-1, 0), x] * _tmp[max(y-1, 0), max(x-1, 0)] * _tmp[y, max(x-1, 0)])
                c += (_tmp[y, max(x-1, 0)] - _tmp[y, max(x-1, 0)] * _tmp[min(y+1, H-1), max(x-1, 0)] * _tmp[min(y+1, H-1), x])
                c += (_tmp[min(y+1, H-1), x] - _tmp[min(y+1, H-1), x] * _tmp[min(y+1, H-1), min(x+1, W-1)] * _tmp[y, min(x+1, W-1)])
                if c == 1:
                    judge += 1
              
                ## condition 3
                if np.sum(tmp[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 3:
                    judge += 1

                ## condition 4
                if np.sum(out[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 2:
                    judge += 1

                ## condition 5
                _tmp2 = 1 - out

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * (1 - tmp[max(y-1, 0), max(x-1, 0)]) * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), max(x-1,0 )] != tmp[max(y-1, 0), max(x-1, 0)]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * (1 - tmp[max(y-1, 0), x]))
                c += ((1-tmp[max(y-1, 0), x]) - (1 - tmp[max(y-1, 0), x]) * _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1,0 )] - _tmp2[y, max(x-1,0 )] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), x] != tmp[max(y-1, 0), x]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * (1 - tmp[max(y-1, 0), min(x+1, W-1)]) * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), min(x+1, W-1)] != tmp[max(y-1, 0), min(x+1, W-1)]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * _tmp2[max(y-1, 0), max(x-1, 0)] * (1 - tmp[y, max(x-1, 0)]))
                c += ((1 - tmp[y, max(x-1, 0)]) - (1 - tmp[y, max(x-1, 0)]) * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[y, max(x-1, 0)] != tmp[y, max(x-1, 0)]):
                    judge += 1
          
                if judge >= 8:
                    out[y, x] = 0
                    count += 1
              
    out = out.astype(np.uint8) * 255

    return out


# Read image
img = cv2.imread("gazo.png").astype(np.float32)

# hilditch thining
out = hilditch(img)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 65.Zhang-Suen细化算法

将 `gazo.png`进行Zhang-Suen细化算法处理吧！

但是，请注意，有必要反转 `gazo.png`的值，因为以下所有操作都将0作为线，将1作为背景。

对于中心像素$x_1(x,y)$的$8-$近邻定义如下：

$$
\begin{matrix}
x_9&x_2&x_3\\
x_8&x_1&x_4\\
x_7&x_6&x_5
\end{matrix}
$$

考虑以下两个步骤：

* 步骤一：执行光栅扫描并标记满足以下5个条件的所有像素：

  1. 这是一个黑色像素；
  2. 顺时针查看$x_2$、$x_3$、$\cdots$、$x_9$、$x_2$时，从$0$到$1$的变化次数仅为$1$；
  3. $x_2$、$x_3$、$\cdots$、$x_9$中$1$的个数在$2$个以上$6$个以下；
  4. $x_2$、$x_4$、$x_6$中的一个为1；
  5. $x_4$、$x_6$、$x_8$中的一个为1；

  将标记的像素全部变为$1$。
* 步骤二：执行光栅扫描并标记满足以下5个条件的所有像素：

  1. 这是一个黑色像素；
  2. 顺时针查看$x_2$、$x_3$、$\cdots$、$x_9$、$x_2$时，从0到1的变化次数仅为1；
  3. $x_2$、$x_3$、$\cdots$、$x_9$中$1$的个数在$2$个以上$6$个以下；
  4. $x_2$、$x_4$、$x_6$中的一个为1；
  5. $x_2$、$x_6$、$x_8$中的一个为1；

  将标记的像素全部变为$1$。

反复执行步骤一和步骤二直到没有点变化。[^0]

```Python
# Zhang Suen thining algorythm
def Zhang_Suen_thining(img):
    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H-1):
            for x in range(1, W-1):
          
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
              
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue
          
                # condition 4
                if out[y-1, x] + out[y, x+1] + out[y+1, x] < 1:
                    continue

                # condition 5
                if out[y, x+1] + out[y+1, x] + out[y, x-1] < 1:
                    continue
              
                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H-1):
            for x in range(1, W-1):
          
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
              
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue
          
                # condition 4
                if out[y-1, x] + out[y, x+1] + out[y, x-1] < 1:
                    continue

                # condition 5
                if out[y-1, x] + out[y+1, x] + out[y, x-1] < 1:
                    continue
              
                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out


# Read image
img = cv2.imread("gazo.png").astype(np.float32)

# Zhang Suen thining
out = Zhang_Suen_thining(img)


# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 66.方向梯度直方图（HOG）第一步：梯度幅值・梯度方向

HOG（Histogram of Oriented Gradients）是一种表示图像特征量的方法。特征量是表示图像的状态等的向量集合。

在图像识别（图像是什么）和检测（物体在图像中的哪个位置）中，我们需要：

1. 从图像中获取特征量（特征提取）；
2. 基于特征量识别和检测（识别和检测）。

由于深度学习通过机器学习自动执行特征提取和识别，所以看不到 HOG，但在深度学习变得流行之前，HOG 经常被用作特征量表达。

通过以下算法获得HOG：

1. 图像灰度化之后，在x方向和y方向上求出亮度的梯度：
2. 从$g_x$和$g_y$确定梯度幅值和梯度方向：
3. 将梯度方向$[0,180]$进行9等分量化。也就是说，对于$[0,20]$量化为 index 0，对于$[20,40]$量化为 index 1……，角度决定放到哪个格子，实际放的是大小，这个图稍微有点问题，并不是严格按照区间放的
   `<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%A2%AF%E5%BA%A6%E7%9B%B4%E6%96%B9%E5%9B%BE%E9%87%8F%E5%8C%96.PNG" alt=" " style="zoom: 67%;" />`

   `<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E7%BD%91%E6%A0%BC%E7%9B%B4%E6%96%B9%E5%9B%BE.PNG" style="zoom:50%;" />`
4. 将图像划分为$N \times N$个区域（该区域称为 cell），并作出 cell 内步骤3得到的 index 的直方图。 然而，该显示不是1，而是梯度角度。

   <img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%9B%BE%E7%89%87%E5%88%92%E5%88%86%E4%B8%BA%E7%BD%91%E6%A0%BC.PNG" style="zoom:67%;" />
5. C x  C个 cell 被称为一个 block。对每个 block 内的 cell 的直方图通过下面的式子进行归一化。由于归一化过程中窗口一次移动一个 cell 来完成的，因此一个 cell 会被归一化多次，通常$\epsilon=1$：

   $$
   h(t)=\frac{h(t)}{\sqrt{\sum\ h(t)+\epsilon}}
   $$

![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%9B%BE%E5%9D%97%E5%BD%92%E4%B8%80%E5%8C%96.gif)

以上，求出 HOG 特征值。

这一问，我们完成步骤1到3。

为了使示例答案更容易看出效果，`gra`是彩色的。此外，`mag`被归一化至$[0,255]$。

求出 `imori.jpg`的 HOG 特征量的梯度幅值和梯度方向吧！

|                        输入 (imori.jpg)                         |                   梯度幅值(answers/answer_66_mag.jpg)                   |                   梯度方向(answers/answer_66_gra.jpg)                   |
| :-------------------------------------------------------------: | :---------------------------------------------------------------------: | :---------------------------------------------------------------------: |
| ![](https://img.geek-docs.com/opencv/opencv-examples/imori.jpg) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_66_mag.jpg) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_66_gra.jpg) |

```Python
# get HOG step1
def HOG_step1(img):
     # Grayscale
     def BGR2GRAY(img):
          gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
          return gray

     # Magnitude and gradient  求x与y方向的梯度
     def get_gradXY(gray):
          H, W = gray.shape

          # padding before grad
          gray = np.pad(gray, (1, 1), 'edge')

          # get grad x
          gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
          # get grad y
          gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
          # replace 0 with 
          gx[gx == 0] = 1e-6

          return gx, gy

     # get magnitude and gradient  求梯度的大小和角度
     def get_MagGrad(gx, gy):
          # get gradient maginitude
          magnitude = np.sqrt(gx ** 2 + gy ** 2)

          # get gradient angle
          gradient = np.arctan(gy / gx)

          gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

          return magnitude, gradient

     # Gradient histogram  对角度进行量化
     def quantization(gradient):
          # prepare quantization table
          gradient_quantized = np.zeros_like(gradient, dtype=np.int)

          # quantization base
          d = np.pi / 9

          # quantization
          for i in range(9):
               gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

          return gradient_quantized

     # 1. BGR -> Gray 灰度化
     gray = BGR2GRAY(img)

     # 1. Gray -> Gradient x and y 灰度图求x方向和y方向的梯度
     gx, gy = get_gradXY(gray)

     # 2. get gradient magnitude and angle  得到梯度大小和角度
     magnitude, gradient = get_MagGrad(gx, gy)

     # 3. Quantization 对梯度角度进行量化
     gradient_quantized = quantization(gradient)

     return magnitude, gradient_quantized   #返回梯度的大小和量化后的梯度角度


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG step1
magnitude, gradient_quantized = HOG_step1(img)

# Write gradient magnitude to file
_magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

cv2.imwrite("out_mag.jpg", _magnitude)

# Write gradient angle to file
H, W, C = img.shape
out = np.zeros((H, W, 3), dtype=np.uint8)

# define color
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
     [127, 127, 0], [127, 0, 127], [0, 127, 127]]

# draw color
for i in range(9):
     out[gradient_quantized == i] = C[i]


cv2.imwrite("out_gra.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 67.方向梯度直方图（HOG）第二步：梯度直方图

在这里完成 HOG 的第4步。

取$N=8$，$8 \times 8$个像素为一个 cell，将每个 cell 的梯度幅值加到梯度方向的index处。[^2]

解答为按照下面的顺序排列索引对应的直方图：

$$
\begin{matrix}
1&2& 3\\
4& 5& 6\\
7& 8 &9
\end{matrix}
$$

就是将上面的直方图按照这样的顺序排列

|                                  输入 (imori.jpg)                                   |                     输出(answers/answer_67.png)                     |
| :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_67.png) |

```Python
# get HOG step2
def HOG_step2(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized

  
    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)

    return histogram


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG step2
histogram = HOG_step2(img)
          
# write histogram to file
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(histogram[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
plt.savefig("out.png")
plt.show()
```

### 68.方向梯度直方图（HOG）第三步：直方图归一化

在这里完成 HOG 的第5步。

取$C=3$，将$3\times 3$个 cell 看作一个 block，进行直方图归一化，通常$\epsilon=1$：

$$
h(t)=\frac{h(t)}{\sqrt{\sum\ h(t)+\epsilon}}
$$

在此，我们得到HOG特征量。

|                                  输入 (imori.jpg)                                   |                                输出(answers/answer_67.png)                                 |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | `<img src="https://img.geek-docs.com/opencv/opencv-examples/answer_68.png" width="400px">` |

```Python
# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
  
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG
histogram = HOG(img)
          
# Write result to file
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(histogram[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
plt.savefig("out.png")
plt.show()

```

### 69.方向梯度直方图（HOG）第四步：可视化特征量

在这里我们将得到的特征量可视化。

如果将特征量叠加在灰度化后的 `imori.jpg`上，可以很容易看到（蝾螈的）外形。

一个好的可视化的方法是这样的，为 cell 内的每个 index 的方向画一条线段，并且值越大，线段越白，值越小，线段越黑。

解答例

|                                  输入 (imori.jpg)                                   |                     输出(answers/answer_69.jpg)                     |
| :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_69.jpg) |

```Python
# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
  
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram

# draw HOG
def draw_HOG(img, histogram):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    def draw(gray, histogram, N=8):
        # get shape
        H, W = gray.shape
        cell_N_H, cell_N_W, _ = histogram.shape
  
        ## Draw
        out = gray[1 : H + 1, 1 : W + 1].copy().astype(np.uint8)

        for y in range(cell_N_H):
            for x in range(cell_N_W):
                cx = x * N + N // 2
                cy = y * N + N // 2
                x1 = cx + N // 2 - 1
                y1 = cy
                x2 = cx - N // 2 + 1
                y2 = cy
          
                h = histogram[y, x] / np.sum(histogram[y, x])
                h /= h.max()
  
                for c in range(9):
                    #angle = (20 * c + 10 - 90) / 180. * np.pi
                    # get angle
                    angle = (20 * c + 10) / 180. * np.pi
                    rx = int(np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy) + cx)
                    ry = int(np.cos(angle) * (x1 - cx) - np.cos(angle) * (y1 - cy) + cy)
                    lx = int(np.sin(angle) * (x2 - cx) + np.cos(angle) * (y2 - cy) + cx)
                    ly = int(np.cos(angle) * (x2 - cx) - np.cos(angle) * (y2 - cy) + cy)

                    # color is HOG value
                    c = int(255. * h[c])

                    # draw line
                    cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)

        return out
  

    # get gray
    gray = BGR2GRAY(img)

    # draw HOG
    out = draw(gray, histogram)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG
histogram = HOG(img)

# draw HOG
out = draw_HOG(img, histogram)


# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 70.色彩追踪（Color Tracking）

色彩追踪是提取特定颜色的区域的方法。

然而，由于在 RGB 色彩空间内颜色有$256^3$种，因此十分困难（或者说手动提取相当困难），因此进行 HSV 变换。

HSV 变换在问题5中提到过，是将 RGB 变换到色相（Hue）、饱和度（Saturation）、明度（Value）的方法。

- 饱和度越小越白，饱和度越大颜色越浓烈，$0\leq S\leq 1$；
- 明度数值越高越接近白色，数值越低越接近黑色（$0\leq V\leq 1$）；
  | - 色相：将颜色使用0到360度表示，具体色相与数值按下表对应： |  红   |  黄   |  绿   | 青色  | 蓝色  | 品红  | 红 |
  | :--------------------------------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: |
  |                             0°                             |  60°  | 120°  | 180°  | 240°  | 300°  | 360°  |

也就是说，为了追踪蓝色，可以在进行 HSV 转换后提取其中$180\leq H\leq 260$的位置，将其变为$255$。

在HSV色彩空间内对 `imori.jpg`创建一个只有蓝色部分值为255的图像。

|                                  输入 (imori.jpg)                                   |                     输出(answers/answer_70.png)                     |
| :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://img.geek-docs.com/opencv/opencv-examples/answer_70.png) |

```Python
# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()

	return hsv

# make mask
def get_mask(hsv):
	mask = np.zeros_like(hsv[..., 0])
	#mask[np.where((hsv > 180) & (hsv[0] < 260))] = 255
	mask[np.logical_and((hsv[..., 0] > 180), (hsv[..., 0] < 260))] = 255
	return mask


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img)

# color tracking
mask = get_mask(hsv)

out = mask.astype(np.uint8)

```

### 71.掩膜（Masking）

像这样通过使用黑白二值图像将对应于黑色部分的原始图像的像素改变为黑色的操作被称为掩膜。

要提取蓝色部分，请先创建这样的二进制图像，使得 `HSV`色彩空间中$180\leq H\leq 260$的位置的像素值设为1，并将其0和1反转之后与原始图像相乘。

这使得可以在某种程度上将蝾螈（从背景上）分离出来。

使用 `HSV`对 `imori.jpg`进行掩膜处理，只让蓝色的地方变黑。

|                                        原图                                         |                                         掩膜                                         |                                             结果                                              |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | +![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%8E%A9%E8%86%9C.png) | =![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%8E%A9%E8%86%9C%E5%90%8E.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()

	return hsv

# make mask
def get_mask(hsv):
	mask = np.zeros_like(hsv[..., 0])
	#mask[np.where((hsv > 180) & (hsv[0] < 260))] = 255
	mask[np.logical_and((hsv[..., 0] > 180), (hsv[..., 0] < 260))] = 1
	return mask

# masking
def masking(img, mask):
	mask = 1 - mask
	out = img.copy()
	# mask [h, w] -> [h, w, channel]
	mask = np.tile(mask, [3, 1, 1]).transpose([1, 2, 0])
	out *= mask

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img / 255.)

# color tracking
mask = get_mask(hsv)

# masking
out = masking(img, mask)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 72.掩膜（色彩追踪（Color Tracking）+形态学处理）

在问题71中掩膜并不是十分精细，蝾螈的眼睛被去掉，背景也有些许残留。

因此，可以通过对掩膜图像应用 `N = 5`闭运算（问题五十）和开运算（问题四十九），以使掩膜图像准确。

|                                        原图                                         |                                                  掩膜                                                  |                                                  结果                                                  |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E4%BC%98%E5%8C%96%E6%8E%A9%E8%86%9C1.png) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E4%BC%98%E5%8C%96%E7%BB%93%E6%9E%9C1.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()

	return hsv

# make mask 制作最初掩膜
def get_mask(hsv):
	mask = np.zeros_like(hsv[..., 0])
	#mask[np.where((hsv > 180) & (hsv[0] < 260))] = 255
	mask[np.logical_and((hsv[..., 0] > 180), (hsv[..., 0] < 260))] = 1
	return mask

# masking
def masking(img, mask):
	mask = 1 - mask
	out = img.copy()
	# mask [h, w] -> [h, w, channel]
	mask = np.tile(mask, [3, 1, 1]).transpose([1, 2, 0])
	out *= mask

	return out


# Erosion
def Erode(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2 , x - 1 : x + 2]) < 1 * 4:
					out[y - 1, x - 1] = 0

	return out


# Dilation
def Dilate(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2, x - 1 : x + 2]) >= 1:
					out[y - 1, x - 1] = 1

	return out


# Opening morphology
def Morphology_Opening(img, time=1):
    out = Erode(img, Erode_time=time)
    out = Dilate(out, Dil_time=time)
    return out

# Closing morphology
def Morphology_Closing(img, time=1):
	out = Dilate(img, Dil_time=time)
	out = Erode(out, Erode_time=time)
	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img / 255.)

# color tracking
mask = get_mask(hsv)

# closing
mask = Morphology_Closing(mask, time=5)

# opening
mask = Morphology_Opening(mask, time=5)

# masking
out = masking(img, mask)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 73.缩小和放大

放大缩小的时候使用双线性插值。如果将双线性插值方法编写成函数的话，编程会变得简洁一些。

将 `imori.jpg`进行灰度化处理之后，先缩小至原来的$0.5$倍，再放大两倍吧。这样做的话，会得到模糊的图像。

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Bi-Linear interpolation 双线性插值
def bl_interpolate(img, ax=1., ay=1.):
	if len(img.shape) > 2:
		H, W, C = img.shape
	else:
		H, W = img.shape
		C = 1

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance 
	dx = x - ix
	dy = y - iy

	if C > 1:
		dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
		dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float)

gray = BGR2GRAY(img)

# Bilinear interpolation
out = bl_interpolate(gray.astype(np.float32), ax=0.5, ay=0.5)

# Bilinear interpolation
out = bl_interpolate(out, ax=2., ay=2.)

out = out.astype(np.uint8)
```

### 74.使用差分金字塔提取高频成分

求出问题73中得到的图像与原图像的差，并将其正规化至$[0,255]$范围。

在这里求得的就是图像的边缘。即，图像的高频成分。

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Bi-Linear interpolation
def bl_interpolate(img, ax=1., ay=1.):
	if len(img.shape) > 2:
		H, W, C = img.shape
	else:
		H, W = img.shape
		C = 1

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance 
	dx = x - ix
	dy = y - iy

	if C > 1:
		dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
		dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float)

gray = BGR2GRAY(img)  #灰度化

# Bilinear interpolation   缩小
out = bl_interpolate(gray.astype(np.float32), ax=0.5, ay=0.5)

# Bilinear interpolation   放大
out = bl_interpolate(out, ax=2., ay=2.)

out = np.abs(out - gray)   #求差值得到边缘

out = out / out.max() * 255  #正规化

out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 75.高斯金字塔（Gaussian Pyramid）

在这里我们求出原图像$\frac{1}{2}$, $\frac{1}{4}$, $\frac{1}{8}$, $\frac{1}{16}$, $\frac{1}{32}$大小的图像。

像这样把原图像缩小之后（像金字塔一样）重叠起来的就被称为高斯金字塔。
                               ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E9%AB%98%E6%96%AF%E9%87%91%E5%AD%97%E5%A1%94.png)
这种高斯金字塔的方法现在仍然有效。高斯金字塔的方法也用于提高图像清晰度的超分辨率成像（Super-Resolution ）深度学习方法。

|                                        原图                                         |                                                 灰度图                                                 |                                          1/2                                          |                                          1/4                                          |                                          1/8                                          |                                          1/16                                          |                                          1/32                                          |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88%E7%81%B0%E5%BA%A61.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88_2.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88_4.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88_8.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88_16.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88_32.jpg) |

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Bi-Linear interpolation
def bl_interpolate(img, ax=1., ay=1.):
	if len(img.shape) > 2:
		H, W, C = img.shape
	else:
		H, W = img.shape
		C = 1

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance 
	dx = x - ix
	dy = y - iy

	if C > 1:
		dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
		dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out

# make image pyramid
def make_pyramid(gray):
	# first element
	pyramid = [gray]
	# each scale
	for i in range(1, 6):
		# define scale
		a = 2. ** i

		# down scale
		p = bl_interpolate(gray, ax=1./a, ay=1. / a)

		# add pyramid list
		pyramid.append(p)

	return pyramid

# Read image
img = cv2.imread("imori.jpg").astype(np.float)

gray = BGR2GRAY(img)

# pyramid
pyramid = make_pyramid(gray)

for i in range(6):
    cv2.imwrite("out_{}.jpg".format(2**i), pyramid[i].astype(np.uint8))
    plt.subplot(1, 6, i+1)
    plt.imshow(pyramid[i], cmap='gray')
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")

plt.show()
```

### 76.显著图（Saliency Map）

在这里我们使用高斯金字塔制作简单的显著图。

显著图是将一副图像中容易吸引人的眼睛注意的部分（突出）表现的图像。

虽然现在通常使用深度学习的方法计算显著图，但是一开始人们用图像的 `RGB`成分或者 `HSV`成分创建高斯金字塔，并通过求差来得到显著图（例如[Itti等人的方法](http://ilab.usc.edu/publications/doc/IttiKoch00vr.pdf)）。

在这里我们使用在问题75中得到的高斯金字塔来简单地求出显著图。算法如下：

1. 我们使用双线性插值调整图像大小至$\frac{1}{128}$、 $\frac{1}{64}$、$\frac{1}{32}$……，再使用双线性插值将生成的这些图像放大到原来的大小。。
2. 将得到的金字塔（我们将金字塔的各层分别编号为0,1,2,3,4,5）两两求差。
3. 将第2步中求得的差分全部相加，并正规化至$[0,255]$。

完成以上步骤就可以得到显著图了。虽然第2步中并没有指定要选择哪两张图像，但如果选择两个好的图像，则可以像答案那样得到一张显著图。

从图上可以清楚地看出，蝾螈的眼睛部分和颜色与周围不太一样的地方变成了白色，这些都是人的眼睛容易停留的地方。

解答中使用了$(0,1)$、$(0,3)$、$(0,5)$、$(1,4)$、$(2,3)$、$(3,5)$。

|                                  输入 (imori.jpg)                                   |                             输出(answers/answer_76.jpg)                             |
| :---------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%98%BE%E8%91%97.jpg) |

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Bi-Linear interpolation 双线性插值
def bl_interpolate(img, ax=1., ay=1.):
	if len(img.shape) > 2:
		H, W, C = img.shape
	else:
		H, W = img.shape
		C = 1

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance 
	dx = x - ix
	dy = y - iy

	if C > 1:
		dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
		dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out

# make image pyramid  制作金字塔
def make_pyramid(gray):
	# first element
	pyramid = [gray]
	# each scale
	for i in range(1, 6):
		# define scale
		a = 2. ** i

		# down scale
		p = bl_interpolate(gray, ax=1./a, ay=1. / a)

		# up scale
		p = bl_interpolate(p, ax=a, ay=a)

		# add pyramid list
		pyramid.append(p.astype(np.float32))

	return pyramid

# make saliency map 制作显著图
def saliency_map(pyramid):
	# get shape
	H, W = pyramid[0].shape

	# prepare out image
	out = np.zeros((H, W), dtype=np.float32)

	# add each difference
	out += np.abs(pyramid[0] - pyramid[1])
	out += np.abs(pyramid[0] - pyramid[3])
	out += np.abs(pyramid[0] - pyramid[5])
	out += np.abs(pyramid[1] - pyramid[4])
	out += np.abs(pyramid[2] - pyramid[3])
	out += np.abs(pyramid[3] - pyramid[5])

	# normalization
	out = out / out.max() * 255

	return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# grayscale
gray = BGR2GRAY(img)

# pyramid
pyramid = make_pyramid(gray)
  
# pyramid -> saliency
out = saliency_map(pyramid)

out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
```

### 77.Gabor 滤波器（Gabor Filter）

来进行Gabor 滤波吧。

Gabor 滤波器是一种结合了高斯分布和频率变换的滤波器，用于在图像的特定方向提取边缘。

滤波器由以下式子定义：

其中：

* $x$、$y$是滤波器的位置。滤波器的大小如果为$K$的话，$y$、$x$取$[-k//2,k//2]$；
* $\gamma$：Gabor 滤波器的椭圆度；
* $\sigma$：高斯分布的标准差；
* $\lambda$：波长；
* $p$：相位；
* $A$：滤波核中平行条带的方向。

在这里，取$K=111$，$\sigma=10$，$\gamma = 1.2$，$\lambda =10$，$p=0$，$A=0$，可视化Gabor滤波器吧！

实际使用Gabor滤波器时，通过归一化以使滤波器值的绝对值之和为1使其更易于使用。

在答案中，滤波器值被归一化至$[0,255]$以进行可视化。
                                                                                      ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Gabor%E6%BB%A4%E6%B3%A2%E5%99%A8.jpg)

```Python
# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


# get gabor kernel
gabor = Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0)

# Visualize
# normalize to [0, 255]
out = gabor - np.min(gabor)
out /= np.max(out)
out *= 255

out = out.astype(np.uint8)
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
```

### 78.旋转Gabor滤波器

在这里分别取$A=0,45,90,135$来求得旋转Gabor滤波器。其它参数和问题七十七一样，$K=111$，$\sigma=10$，$\gamma = 1.2$，$\lambda =10$，$p=0$。

Gabor滤波器可以通过这里的方法简单实现。

|                                                        输出                                                         |
| :-----------------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Gabor%E6%97%8B%E8%BD%AC%E6%BB%A4%E6%B3%A2%E5%99%A8.png) |

```Python
# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


# define each angle
As = [0, 45, 90, 135]

# prepare pyplot
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

# each angle
for i, A in enumerate(As):
    # get gabor kernel
    gabor = Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=A)

    # normalize to [0, 255]
    out = gabor - np.min(gabor)
    out /= np.max(out)
    out *= 255
  
    out = out.astype(np.uint8)
    plt.subplot(1, 4, i+1)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.title("Angle "+str(A))

plt.savefig("out.png")
plt.show()
```

### 79.使用Gabor滤波器进行边缘检测

将 `imori.jpg`灰度化之后，分别使用$A=0,45,90,135$的Gabor滤波器进行滤波。其它参数取为：$K=111$，$\sigma=10$，$\gamma = 1.2$，$\lambda =10$，$p=0$。

如在答案示例看到的那样， Gabor滤波器提取了指定的方向上的边缘。因此，Gabor滤波器在边缘特征提取方面非常出色。

一般认为 Gabor 滤波器接近生物大脑视皮层中的初级简单细胞（V1 区）。也就是说，当生物看见眼前的图像时也进行了特征提取。

一般认为深度学习的卷积层接近 Gabor 滤波器的功能。然而，在深度学习中，滤波器的系数通过机器学习自动确定。作为机器学习的结果，据说将发生类似于Gabor滤波器的过程。

|                                  输入 (imori.jpg)                                   |                                              输出(answers/answer_79.png)                                               |
| :---------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | `<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Gabor%E6%BB%A4%E6%B3%A24.png" style="zoom:67%;" />` |

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
  
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def Gabor_process(img):
    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    As = [0, 45, 90, 135]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)

        plt.subplot(1, 4, i+1)
        plt.imshow(out, cmap='gray')
        plt.axis('off')
        plt.title("Angle "+str(A))

    plt.savefig("out.png")
    plt.show()

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# gabor process
Gabor_process(img)
```

### 80.使用Gabor滤波器进行特征提取

通过将问题79中得到的4张图像加在一起，提取图像的特征。

观察得到的结果，图像的轮廓部分是白色的，获得了类似于边缘检测的输出。

深度学习中的卷积神经网络，最初已经具有提取图像的特征的功能，在不断重复特征提取的计算过程中，自动提取图像的特征。

|                                  输入 (imori.jpg)                                   |                               输出(answers/answer_80.jpg)                                |
| :---------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Gabor%E6%BB%A4%E6%B3%A2.jpg) |

```Python
# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor

def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
  
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    As = [0, 45, 90, 135]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# gabor process
out = Gabor_process(img)
```

### 81.Hessian角点检测[^0]

角点检测是检测边缘上的角点。

角点是曲率变大的点，下式定义了高斯曲率：

$$
K=\frac{\det(H)}{(1+{I_x}^2+{I_y}^2)^2}
$$

其中：

* $\det(H)=I_{xx}\ I_{yy}-{I_{xy}}^2$；
* $H$表示Hessian矩阵。图像的二次微分（通过将Sobel滤波器应用于灰度图像计算得来）。对于图像上的一点，按照下式定义：
  * $I_x$：应用$x$方向上的Sobel滤波器；
  * $I_y$：应用$y$方向上的Sobel滤波器；
  * $H=\left[\begin{matrix}I_{xx}&I_{xy}\\I_{xy}&I_{yy}\end{matrix}\right]$

在Hessian角点检测中，$\det{H}$将极大点视为j角点。

如果中心像素与其$8-$近邻像素相比值最大，则中心像素为极大点。

解答中，角点是$\det(H)$为极大值，并且大于$\max(\det(H))\cdot 0.1$的点。
对 `thorino.jpg`进行Hessian角点检测吧！

|                                                           原图                                                           |                                                        角点检测                                                         |
| :----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E5%9B%BE1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E5%9B%BE.jpg) |

```Python
# Hessian corner detection
def Hessian_corner(img):

	## Grayscale
	def BGR2GRAY(img):
		gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
		gray = gray.astype(np.uint8)
		return gray

	## Sobel
	def Sobel_filtering(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)

		Ix2 = Ix ** 2
		Iy2 = Iy ** 2
		Ixy = Ix * Iy

		return Ix2, Iy2, Ixy



	## Hessian
	def corner_detect(gray, Ix2, Iy2, Ixy):
		# get shape
		H, W = gray.shape

		# prepare for show detection
		out = np.array((gray, gray, gray))
		out = np.transpose(out, (1,2,0))

		# get Hessian value
		Hes = np.zeros((H, W))

		for y in range(H):
			for x in range(W):
				Hes[y,x] = Ix2[y,x] * Iy2[y,x] - Ixy[y,x] ** 2

		## Detect Corner and show
		for y in range(H):
			for x in range(W):
				if Hes[y,x] == np.max(Hes[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) and Hes[y, x] > np.max(Hes) * 0.1:
					out[y, x] = [0, 0, 255]

		out = out.astype(np.uint8)

		return out


	# 1. grayscale
	gray = BGR2GRAY(img)

	# 2. get difference image
	Ix2, Iy2, Ixy = Sobel_filtering(gray)

	# 3. corner detection
	out = corner_detect(gray, Ix2, Iy2, Ixy)

	return out


# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Hessian corner detection
out = Hessian_corner(img)

cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
```

### 82.Harris角点检测第一步：Sobel + Gausian

问题82和问题83对 `thorino.jpg`进行 Harris 角点检测

Harris 角点检测算法如下：

1. 对图像进行灰度化处理；
2. 利用Sobel滤波器求出海森矩阵（Hessian matrix）：[^1]

   $$
   H=\left[\begin{matrix}{I_x}^2&I_xI_y\\I_xI_y&{I_y}^2\end{matrix}\right]
   $$
3. 将高斯滤波器分别应用于${I_x}^2$、${I_y}^2$、$I_x\ I_y$；
4. 计算每个像素的$R = \det(H) - k\ (\text{trace}(H))^2$。通常$K$在$[0.04,0.16]$范围内取值.
5. 满足 $R \geq \max(R) \cdot\text{th}  $的像素点即为角点。

问题八十二至问题八十三中的参数如下：

* 高斯滤波器：$k=3, \sigma=3$；
* $K = 0.04, \text{th} = 0.1$。

在这里我们完成步骤1到步骤3。

|                                                           原图                                                           |                                               Sobel + Gausian                                                |
| :----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E5%9B%BE1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Harris%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B1.png) |

```Python
# Harris corner detection
def Harris_corner_step1(img):

	## Grayscale
	def BGR2GRAY(img):
		gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
		gray = gray.astype(np.uint8)
		return gray

	## Sobel
	def Sobel_filtering(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)

		Ix2 = Ix ** 2
		Iy2 = Iy ** 2
		Ixy = Ix * Iy

		return Ix2, Iy2, Ixy


	# gaussian filtering
	def gaussian_filtering(I, K_size=3, sigma=3):
		# get shape
		H, W = I.shape

		## gaussian
		I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

		# gaussian kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(K_size):
			for y in range(K_size):
				_x = x - K_size // 2
				_y = y - K_size // 2
				K[y, x] = np.exp( -(_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
		K /= (sigma * np.sqrt(2 * np.pi))
		K /= K.sum()

		# filtering
		for y in range(H):
			for x in range(W):
				I[y,x] = np.sum(I_t[y : y + K_size, x : x + K_size] * K)
	
		return I


	# 1. grayscale
	gray = BGR2GRAY(img)

	# 2. get difference image
	Ix2, Iy2, Ixy = Sobel_filtering(gray)

	# 3. gaussian filtering
	Ix2 = gaussian_filtering(Ix2, K_size=3, sigma=3)
	Iy2 = gaussian_filtering(Iy2, K_size=3, sigma=3)
	Ixy = gaussian_filtering(Ixy, K_size=3, sigma=3)

	# show result
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

	plt.subplot(1,3,1)
	plt.imshow(Ix2, cmap='gray')
	plt.title("Ix^2")
	plt.axis("off")

	plt.subplot(1,3,2)
	plt.imshow(Iy2, cmap='gray')
	plt.title("Iy^2")
	plt.axis("off")

	plt.subplot(1,3,3)
	plt.imshow(Ixy, cmap='gray')
	plt.title("Ixy")
	plt.axis("off")

	plt.savefig("out.png")
	plt.show()


# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Harris corner detection step1
out = Harris_corner_step1(img)
```

### 83.Harris角点检测第二步：角点检测

在这里进行算法的步骤四和步骤五吧！

在步骤四中，$K = 0.04$；在步骤五中$\text{th} = 0.1$。

|                                                           原图                                                           |                                                   角点检测                                                   |
| :----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E5%9B%BE1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/Harris%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B2.jpg) |

```Python
# Harris corner detection
def Harris_corner(img):

	## Grayscale
	def BGR2GRAY(img):
		gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
		gray = gray.astype(np.uint8)
		return gray

	## Sobel
	def Sobel_filtering(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)

		Ix2 = Ix ** 2
		Iy2 = Iy ** 2
		Ixy = Ix * Iy

		return Ix2, Iy2, Ixy

	# gaussian filtering
	def gaussian_filtering(I, K_size=3, sigma=3):
		# get shape
		H, W = I.shape

		## gaussian
		I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

		# gaussian kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(K_size):
			for y in range(K_size):
				_x = x - K_size // 2
				_y = y - K_size // 2
				K[y, x] = np.exp( -(_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
		K /= (sigma * np.sqrt(2 * np.pi))
		K /= K.sum()

		# filtering
		for y in range(H):
			for x in range(W):
				I[y,x] = np.sum(I_t[y : y + K_size, x : x + K_size] * K)
	
		return I

	# corner detect
	def corner_detect(gray, Ix2, Iy2, Ixy, k=0.04, th=0.1):
		# prepare output image
		out = np.array((gray, gray, gray))
		out = np.transpose(out, (1,2,0))

		# get R
		R = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)

		# detect corner
		out[R >= np.max(R) * th] = [0, 0, 255]

		out = out.astype(np.uint8)

		return out

	# 1. grayscale
	gray = BGR2GRAY(img)

	# 2. get difference image
	Ix2, Iy2, Ixy = Sobel_filtering(gray)

	# 3. gaussian filtering
	Ix2 = gaussian_filtering(Ix2, K_size=3, sigma=3)
	Iy2 = gaussian_filtering(Iy2, K_size=3, sigma=3)
	Ixy = gaussian_filtering(Ixy, K_size=3, sigma=3)

	# 4. corner detect
	out = corner_detect(gray, Ix2, Iy2, Ixy)

	return out

# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Harris corner detection
out = Harris_corner(img)

cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
```

### 84.简单图像识别第一步：减色化+柱状图[^3]

这里我们进行简单的图像识别。

图像识别是识别图像中物体的类别（它属于哪个类）的任务。图像识别通常被称为Classification、Categorization、Clustering等。

一种常见的方法是通过 HOG、SIFT、SURF 等方法从图像中提取一些特征，并通过特征确定物体类别。这种方法在CNN普及之前广泛采用，但CNN可以完成从特征提取到分类等一系列任务。

这里，利用图像的颜色直方图来执行简单的图像识别。算法如下：

1. 将图像 `train_***.jpg`进行减色处理（像问题六中那样，$\text{RGB}$取4种值）。
2. 创建减色图像的直方图。直方图中，$\text{RGB}$分别取四个值，但为了区分它们，$B = [1,4]$、$G = [5,8]$、$R = [9,12]$，这样$bin=12$。请注意，我们还需要为每个图像保存相应的柱状图。也就是说，需要将数据储存在 `database = np.zeros((10(训练数据集数), 13(RGB + class), dtype=np.int)`中。
3. 将步骤2中计算得到的柱状图记为 database。
4. 计算想要识别的图像 `test@@@.jpg`与直方图之间的差，将差称作特征量。
5. 直方图的差异的总和是最小图像是预测的类别。换句话说，它被认为与近色图像属于同一类。
6. 计算将想要识别的图像（`test_@@@.jpg`）的柱状图（与 `train_***.jpg`的柱状图）的差，将这个差作为特征量。
7. 统计柱状图的差，差最小的图像为预测的类别。换句话说，可以认为待识别图像与具有相似颜色的图像属于同一类。

在这里，实现步骤1至步骤3并可视化柱状图。

训练数据集存放在文件夹 `dataset`中，分为 `trainakahara@@@.jpg`（类别1）和 `trainmadara@@@.jpg`（类别2）两类，共计10张。`akahara`是红腹蝾螈（Cynops pyrrhogaster），`madara`是理纹欧螈（Triturus marmoratus）。

这种预先将特征量存储在数据库中的方法是第一代人工智能方法。这个想法是逻辑是，如果你预先记住整个模式，那么在识别的时候就没有问题。但是，这样做会消耗大量内存，这是一种有局限的方法。

|                                                               输出                                                               |
| :------------------------------------------------------------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E7%9B%B4%E6%96%B9%E5%9B%BE.png) |

```bash
被存储的直方图的内容
[[  172 12254  2983   975   485 11576  3395   928   387 10090  4845  1062  0]
[ 3627  7350  4420   987  1743  8438  4651  1552   848  9089  4979  1468  0]
[ 1646  6547  5807  2384  1715  8502  5233   934  1553  5270  7167  2394  0]
[  749 10142  5465    28  1431  7922  7001    30  1492  7819  7024    49  0]
[  927  4197  8581  2679   669  5689  7959  2067   506  3973  6387  5518  0]
[ 2821  6404  2540  4619  1625  7317  3019  4423   225  8635  1591  5933  1]
[ 5575  7831  1619  1359  4638  6777  3553  1416  4675  7964  2176  1569  1]
[ 4867  7523  3275   719  4457  6390  3049  2488  4328  7135  3377  1544  1]
[ 7881  6160  1992   351  7426  3967  4258   733  7359  4979  3322   724  1]
[ 5638  6580  3916   250  5041  4185  6286   872  5226  4930  5552   676  1]]
```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

## Dicrease color 减色处理
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

## Database
def get_DB():
    # get image paths
    train = glob("dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)

    # each image
    for i, path in enumerate(train):
        img = dic_color(cv2.imread(path))
        # get histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        img_h = img.copy() // 64
        img_h[..., 1] += 4
        img_h[..., 2] += 8
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path)

    print(db)
    plt.show()

# get database
get_DB()
```

### 85.简单图像识别第二步：判别类别

在这里我们完成算法的4至5步。

请使用测试数据集 `testakahara@@@.jpg`和 `testmadara@@@.jpg`（共计4张）。请输出各个与各个图像直方图差别最小的（训练数据集的）文件名和预测类别。这种评价方法被称为最近邻法（Neareset Neighbour）。

答案如下：

```bash
test_akahara_1.jpg is similar >> train_akahara_3.jpg  Pred >> akahara
test_akahara_2.jpg is similar >> train_akahara_1.jpg  Pred >> akahara
test_madara_1.jpg is similar >> train_madara_2.jpg  Pred >> madara
test_madara_2.jpg is similar >> train_akahara_2.jpg  Pred >> akahara
```

```Python
# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

# Database
def get_DB():
    # get training image path
    train = glob("dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)

    # prepare path database
    pdb = []

    # each image
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))

        #get histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        # store image path
        pdb.append(path)

    return db, pdb

# test
def test_DB(db, pdb):
    # get test image path
    test = glob("dataset/test_*")
    test.sort()

    success_num = 0.

    # each image
    for path in test:
        # read image
        img = dic_color(cv2.imread(path))

        # get histogram
        hist = np.zeros(12, dtype=np.int32)
        for j in range(4):
            hist[j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            hist[j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            hist[j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get histogram difference
        difs = np.abs(db[:, :12] - hist)
        difs = np.sum(difs, axis=1)

        # get argmin of difference
        pred_i = np.argmin(difs)

        # get prediction label
        pred = db[pred_i, -1]

        if pred == 0:
            pl = "akahara"
        elif pred == 1:
            pl = "madara"
  
        print(path, "is similar >>", pdb[pred_i], " Pred >>", pl)

db, pdb = get_DB()
test_DB(db, pdb)
```

### 86.简单图像识别第三步：评估

在这里对图像识别的结果做评估。

正确率（Accuracy, Precision）用来表示多大程度上分类正确，在图像识别任务上是一般性的评价指标。正确率通过下式计算。要はテストにおける得点率である。当得到的值有小数时，也可以用百分比表示。

$$
\text{Accuracy}=\frac{\text{被正确识别的图像个数}}{\text{图像总数}}
$$

按照上面的方法，求出问题85中的正确率吧！答案如下：

```bash
Accuracy >> 0.75 (3/4)
```

### 87.简单图像识别第四步：k-NN

问题八十五中虽然我们预测了颜色最接近的图像，但实际上和 `testmadara2.jpg`最接近的是 `trainakahara2.jpg`。

如果比较这两个图像，它们绿色和黑色比例看起来差不多，因此整个图像颜色看起来相同。这是因为在识别的时候，训练图像选择了一张偏离大部分情况的图像。因此，训练数据集的特征不能很好地分离，并且有时包括偏离特征分布的样本。

为了避免这中情况发生，在这里我们选择颜色相近的三副图像，并通过投票来预测最后的类别，再计算正确率。

像这样选择具有相似特征的3个学习数据的方法被称为 k-近邻算法（k-NN: k-Nearest Neighbor）。 问题85中的NN 方法是 k = 1 的情况。

```Python
# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

# Database
def get_DB():
    # get training image path
    train = glob("dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)
    pdb = []

    # each train
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))
        # histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        # add image path
        pdb.append(path)

    return db, pdb

# test
def test_DB(db, pdb, N=3):
    # get test image path
    test = glob("dataset/test_*")
    test.sort()

    accuracy_N = 0.

    # each image
    for path in test:
        # read image
        img = dic_color(cv2.imread(path))

        # get histogram
        hist = np.zeros(12, dtype=np.int32)
        for j in range(4):
            hist[j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            hist[j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            hist[j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get histogram difference
        difs = np.abs(db[:, :12] - hist)
        difs = np.sum(difs, axis=1)

        # get top N
        pred_i = np.argsort(difs)[:N]

        # predict class index
        pred = db[pred_i, -1]

        # get class label
        if len(pred[pred == 0]) > len(pred[pred == 1]):
            pl = "akahara"
        else:
            pl = 'madara'

        print(path, "is similar >> ", end='')
        for i in pred_i:
            print(pdb[i], end=', ')
        print("|Pred >>", pl)

        # count accuracy
        gt = "akahara" if "akahara" in path else "madara"
        if gt == pl:
            accuracy_N += 1.

    accuracy = accuracy_N / len(test)
    print("Accuracy >>", accuracy, "({}/{})".format(int(accuracy_N), len(test)))

db, pdb = get_DB()
test_DB(db, pdb)
```

### 88.k-平均聚类算法（k -means Clustering）第一步：生成质心

问题84至问题87的图像识别任务是需要预期输出的简单监督学习（supervised-training）中的一种简单情况。在这里我们通过不需要预期输出的无监督学习（unsupervised-training）来进行图像分类。

最简单的方法是 k-平均聚类算法（k -means Clustering）。

k-平均聚类算法在类别数已知时使用。在质心不断明确的过程中完成特征量的分类任务。
                                  ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%81%9A%E7%B1%BB.png)
k-平均聚类算法如下：

1. 为每个数据随机分配类；
2. 计算每个类的重心；
3. 计算每个数据与重心之间的距离，将该数据分到重心距离最近的那一类；
4. 重复步骤2和步骤3直到没有数据的类别再改变为止。

在这里，以减色化和直方图作为特征量来执行以下的算法：

1. 对图像进行减色化处理，然后计算直方图，将其用作特征量；
2. 对每张图像随机分配类别0或类别1（在这里，类别数为2，以 `np.random.seed (1)`作为随机种子生成器。当 `np.random.random`小于 `th`时，分配类别0；当 `np.random.random`大于等于 `th`时，分配类别1，在这里 `th=0.5`）；
3. 分别计算类别0和类别1的特征量的质心（质心存储在 `gs = np.zeros((Class, 12), dtype=np.float32)`中）；
4. 对于每个图像，计算特征量与质心之间的距离（在此取欧氏距离），并将图像指定为质心更接近的类别。
5. 重复步骤3和步骤4直到没有数据的类别再改变为止。

在这里，实现步骤1至步骤3吧（步骤4和步骤5的循环不用实现）！将图像 `test@@@.jpg`进行聚类。

答案：

```bash
assigned label
[[ 1493  7892  4900  2099  1828  9127  4534   895  1554  6750  5406  2674 0]
[  242 10338  3628  2176   587 12212  2247  1338   434 10822  4506   622 1]
[ 6421  5478   719  3766  5482  4294  2537  4071  5609  4823  2051  3901 0]
[ 3343  8134  4756   151  3787  7588  3935  1074  3595  8444  4069   276 0]]
Grabity
[[ 3752.3333  7168.      3458.3333  2005.3334  3699.      7003.
3668.6667  2013.3334  3586.      6672.3335  3842.      2283.6667]
[  242.     10338.      3628.      2176.       587.     12212.
2247.      1338.       434.     10822.      4506.       622.    ]]
```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

# Database
def get_DB():
    # get training image path
    train = glob("dataset/test_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)
    pdb = []

    # each train
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))
        # histogram
        for j in range(4):## 将直方图数据写入数组
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label 数组最后一列写入分类
        db[i, -1] = cls 

        # add image path
        pdb.append(path)

    return db, pdb

# k-Means step1
def k_means_step1(db, pdb, Class=2):
    # copy database
    feats = db.copy()

    # initiate random seed
    np.random.seed(1)

    # assign random class #随机设置类
    for i in range(len(feats)):
        if np.random.random() < 0.5:
            feats[i, -1] = 0 
        else:
            feats[i, -1] = 1

    # prepare gravity
    gs = np.zeros((Class, 12), dtype=np.float32)
  
    # get gravity 计算质心
    for i in range(Class):
        gs[i] = np.mean(feats[np.where(feats[..., -1] == i)[0], :12], axis=0)
    print("assigned label")
    print(feats)
    print("Grabity")
    print(gs)

db, pdb = get_DB()
k_means_step1(db, pdb)
```

### 89.k-平均聚类算法（k -means Clustering）第二步：聚类（Clustering）

在这里完成算法的步骤4和步骤5，进行聚类吧！

在这里预测类别为0和1，但顺序与问题85至87不同。

因此，k-平均聚类算法是一种完全按范围划分类别的方法。一条数据最后被划分到什么类别只有到最后才清楚。此外，必须预先知道类别的数量。

需要注意的是，k-平均聚类算法最初分配的类别对最后的结果有很大的影响。并且，数据量小的情况下极有可能失败。也就是说，数据量越大最后得到的数据分布越准确。

答案：

```bash
test_akahara_1.jpg  Pred: 0
test_akahara_2.jpg  Pred: 1
test_madara_1.jpg  Pred: 0
test_madara_2.jpg  Pred: 0
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img


# Database
def get_DB():
    # get training image path
    train = glob("dataset/test_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)
    pdb = []

    # each train
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))
        # histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        # add image path
        pdb.append(path)

    return db, pdb

# k-Means step2
def k_means_step2(db, pdb, Class=2):
    # copy database
    feats = db.copy()

    # initiate random seed
    np.random.seed(1)

    # assign random class 
    for i in range(len(feats)):
        if np.random.random() < 0.5:
            feats[i, -1] = 0
        else:
            feats[i, -1] = 1

    while True:
        # prepare greavity
        gs = np.zeros((Class, 12), dtype=np.float32)
        change_count = 0

        # compute gravity
        for i in range(Class):
            gs[i] = np.mean(feats[np.where(feats[..., -1] == i)[0], :12], axis=0)

        # re-labeling
        for i in range(len(feats)):
            # get distance each nearest graviry
            dis = np.sqrt(np.sum(np.square(np.abs(gs - feats[i, :12])), axis=1))

            # get new label
            pred = np.argmin(dis, axis=0)

            # if label is difference from old label
            if int(feats[i, -1]) != pred:
                change_count += 1
                feats[i, -1] = pred

        if change_count < 1:
            break

    for i in range(db.shape[0]):
        print(pdb[i], " Pred:", feats[i, -1])


db, pdb = get_DB()
k_means_step2(db, pdb)
```

### 90.k-平均聚类算法（k -means Clustering）第三步：调整初期类别

使用k-平均聚类算法将8张 `train@@@.jpg`完美地聚类吧！

在这里，通过变更 `np.random.seed()`的值和 `np.random.random() < th`中分割类别的阈值 `th`来更好地预测图片的类别吧！由于 `train@@@.jpg`的图像数量是问题89的两倍，因此可以更容易地聚类。

这只能通过反复试验来完成。

答案：

```bash
train_akahara_1.jpg  Pred: 1
train_akahara_2.jpg  Pred: 1
train_akahara_3.jpg  Pred: 1
train_akahara_4.jpg  Pred: 1
train_akahara_5.jpg  Pred: 1
train_madara_1.jpg  Pred: 0
train_madara_2.jpg  Pred: 0
train_madara_3.jpg  Pred: 0
train_madara_4.jpg  Pred: 0
train_madara_5.jpg  Pred: 0
```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Dicrease color
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

# Database
def get_DB():
    # get training image path
    train = glob("dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)
    pdb = []

    # each train
    for i, path in enumerate(train):
        # read image
        img = dic_color(cv2.imread(path))
        # histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls

        # add image path
        pdb.append(path)

    return db, pdb

# k-Means
def k_means(db, pdb, Class=2, th=0.5):
    # copy database
    feats = db.copy()

    # initiate random seed
    np.random.seed(4)

    # assign random class 
    for i in range(len(feats)):
        if np.random.random() < th:
            feats[i, -1] = 0
        else:
            feats[i, -1] = 1

    while True:
        # prepare greavity
        gs = np.zeros((Class, 12), dtype=np.float32)
        change_count = 0

        # compute gravity
        for i in range(Class):
            gs[i] = np.mean(feats[np.where(feats[..., -1] == i)[0], :12], axis=0)

        # re-labeling
        for i in range(len(feats)):
            # get distance each nearest graviry
            dis = np.sqrt(np.sum(np.square(np.abs(gs - feats[i, :12])), axis=1))

            # get new label
            pred = np.argmin(dis, axis=0)

            # if label is difference from old label
            if int(feats[i, -1]) != pred:
                change_count += 1
                feats[i, -1] = pred

        if change_count < 1:
            break

    for i in range(db.shape[0]):
        print(pdb[i], " Pred:", feats[i, -1])


db, pdb = get_DB()
k_means(db, pdb, th=0.3)

```

### 91.$k-$平均聚类算法进行减色处理第一步----按颜色距离分类

在问题六中涉及到了减色处理，但是在问题六中事先确定了要减少的颜色。这里，$k-$平均聚类算法用于动态确定要减少的颜色。

算法如下：

1. 从图像中随机选取$K$个$\text{RGB}$分量（这我们称作类别）。
2. 将图像中的像素分别分到颜色距离最短的那个类别的索引中去，色彩距离按照下面的方法计算：

   $$
   \text{dis}=\sqrt{(R-R')^2+(G-G')^2+(B-B')^2}
   $$
3. 计算各个索引下像素的颜色的平均值，这个平均值成为新的类别；
4. 如果原来的类别和新的类别完全一样的话，算法结束。如果不一样的话，重复步骤2和步骤3；
5. 将原图像的各个像素分配到色彩距离最小的那个类别中去。

完成步骤1和步骤2。

- 类别数$K=5$；
- 使用 `reshape((HW, 3))`来改变图像大小之后图像将更容易处理；
- 步骤1中，对于 `np.random.seed(0)`，使用 `np.random.choice(np.arrange(图像的HW), 5, replace=False)`；
- 现在先不考虑步骤3到步骤5的循环。

```bash
# 最初选择的颜色
[[140. 121. 148.]
 [135. 109. 122.]
 [211. 189. 213.]
 [135.  86.  84.]
 [118.  99.  96.]]
```

用与最初被选择的颜色的颜色的距离进行分类的索引(算法2)。
答案是0-4，索引值为x50，便于查看。

对 `imori.jpg`利用 $k-$平均聚类算法进行减色处理。

|                                        输入                                         |                                    输出                                    |
| :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E8%9D%BE%E8%9E%88.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_91.jpg) |

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# K-means step1
def k_means_step1(img, Class=5):
	#  get shape
	H, W, C = img.shape

	# initiate random seed
	np.random.seed(0)

	# reshape
	img = np.reshape(img, (H * W, -1))

	# select one index randomly
	i = np.random.choice(np.arange(H * W), Class, replace=False)
	Cs = img[i].copy()

	print(Cs)

	clss = np.zeros((H * W), dtype=int)

	# each pixel
	for i in range(H * W):
		# get distance from base pixel
		dis = np.sqrt(np.sum((Cs - img[i]) ** 2, axis=1))
		# get argmin distance
		clss[i] = np.argmin(dis)

	# show
	out = np.reshape(clss, (H, W)) * 50
	out = out.astype(np.uint8)

	return out


# read image
img = cv2.imread("imori.jpg").astype(np.float32)

# K-means step2
out = k_means_step1(img)

cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
```

### 92.利用 $k-$平均聚类算法进行减色处理第二步----减色处理

实现算法的第3到5步。

```bash
# 选择的颜色
[[182.86730957 156.13246155 180.24510193]
 [156.75152588 123.88993835 137.39085388]
 [227.31060791 199.93135071 209.36465454]
 [ 91.9105835   57.94448471  58.26378632]
 [121.8759613   88.4736557   96.99688721]]
```

减色处理可以将图像处理成手绘风格。如果$k=10$，则可以在保持一些颜色的同时将图片处理成手绘风格。

现在，$k=5$的情况下试着将 `madara.jpg`进行减色处理。

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def k_means(img, Class=5):
    # get shape
    H, W, C = img.shape

    # initiate random seed
    np.random.seed(0)

    # reshape image
    img = np.reshape(img, (H * W, -1))

    # get index randomly
    i = np.random.choice(np.arange(H * W), Class, replace=False)
    Cs = img[i].copy()

    while True:
        # prepare pixel class label
        clss = np.zeros((H * W), dtype=int)
  
        # each pixel
        for i in range(H * W):
            # get distance from index pixel
            dis = np.sqrt(np.sum((Cs - img[i])**2, axis=1))
            # get argmin distance
            clss[i] = np.argmin(dis)

        # selected pixel values
        Cs_tmp = np.zeros((Class, 3))
  
        # each class label
        for i in range(Class):
            Cs_tmp[i] = np.mean(img[clss == i], axis=0)

        # if not any change
        if (Cs == Cs_tmp).all():
            break
        else:
            Cs = Cs_tmp.copy()

    # prepare out image
    out = np.zeros((H * W, 3), dtype=np.float32)

    # assign selected pixel values  
    for i in range(Class):
        out[clss == i] = Cs[i]

    print(Cs)
  
    out = np.clip(out, 0, 255)

    # reshape out image
    out = np.reshape(out, (H, W, 3))
    out = out.astype(np.uint8)

    return out

# read image
img = cv2.imread("imori.jpg").astype(np.float32)

# K-means
out = k_means(img)

cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
```

### 93.准备机器学习的训练数据第一步——计算$\text{IoU}$

从这里开始我们准备机器学习用的训练数据。

我的最终目标是创建一个能够判断图像是否是蝾螈的脸的判别器。因此，我们需要蝾螈的脸部图像和非蝾螈脸部的图像。我们需要编写程序来准备这样的图像。

为此，有必要从单个图像中用矩形框出蝾螈头部（即Ground-truth），如果随机切割的矩形与Ground-truth在一定程度上重合，那么这个矩形框处就是蝾螈的头。

重合程度通过检测评价函数$\text{IoU}$（Intersection over Union）来判断。通过下式进行计算：

$$
\text{IoU}=\frac{|\text{Rol}|}{|R_1 + R_2 - \text{Rol}|}
$$

其中：

* $R_1$：Ground-truth的范围；
* $R_2$：随机框出来的矩形的范围；
* $\text{Rol}$：$R_1$和$R_2$重合的范围。

计算以下两个矩形的$\text{IoU}$吧！

```python
# [x1, y1, x2, y2] x1,y1...矩形左上的坐标  x2,y2...矩形右下的坐标
a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)
```

答案

```bash
0.627907
```

```Python
# get IoU overlap ratio
def iou(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU 左侧边的大值
    iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU  上侧边的大值
    iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU  右侧边的小值
    iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU  下侧边的小值
    iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
    iou_w = iou_x2 - iou_x1
	# get height of IoU
    iou_h = iou_y2 - iou_y1

	# get area of IoU
    area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

# [x1, y1, x2, y2]
a = np.array((50, 50, 150, 150), dtype=np.float32)

b = np.array((60, 60, 170, 160), dtype=np.float32)

print(iou(a, b))

```

### 94.准备机器学习的训练数据第二步——随机裁剪（Random Cropping）

这里，从图像中随机切出200个$60\times60$的矩形。

并且，满足下面的条件：

1. 使用 `np.random.seed(0)`，求出裁剪的矩形的左上角座标 `x1 = np.random.randint(W-60)`和 `y1=np.random.randint(H-60)`；
2. 如果和 Ground-truth （`gt = np.array((47, 41, 129, 103), dtype=np.float32)`）的$\text{IoU}$大于$0.5$，那么就打上标注$1$，小于$0.5$就打上标注$0$。

答案中，标注$1$的矩形用红色画出，标注$0$的矩形用蓝色的线画出，Ground-truth用绿色的线画出。我们简单地准备蝾螈头部和不是头部的图像。

下面，通过从 `imori1.jpg`中随机裁剪图像制作训练数据。

|                            输入 (imori_1.jpg)                            |                        输出(answers/answer_94.jpg)                         |
| :----------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_1.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_94.jpg) |

```Python
import cv2
import numpy as np

np.random.seed(0)

# get IoU overlap ratio
def iou(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU
	iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU
	iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU
	iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU
	iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
	iou_w = iou_x2 - iou_x1
	# get height of IoU
	iou_h = iou_y2 - iou_y1

	# get area of IoU
	area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
	iou = area_iou / (area_a + area_b - area_iou)

	return iou


# crop and create database
def crop_bbox(img, gt, Crop_N=200, L=60, th=0.5):
    # get shape
    H, W, C = img.shape

    # each crop
    for i in range(Crop_N):
        # get left top x of crop bounding box
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L
  
        # crop bounding box
        crop = np.array((x1, y1, x2, y2))
  
        # get IoU between crop box and gt
        _iou = iou(gt, crop)
  
        # assign label
        if _iou >= th:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            label = 1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            label = 0
  
    return img

# read image
img = cv2.imread("imori_1.jpg")

# gt bounding box 边界框
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# get crop bounding box
img = crop_bbox(img, gt)

# draw gt 画框
cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 1)

cv2.imwrite("out.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0)
```

### 95.神经网络（Neural Network）第一步——深度学习（Deep Learning）

将神经网络作为识别器，这就是现在流行的深度学习。

下面的代码是包含输入层、中间层（Unit 数：64）、输出层（1）的网络。这是实现异或逻辑的网络。网络代码参照了[这里](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)：

```python
import numpy as np

np.random.seed(0)

# neural network
class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        # layer 1 weight
        self.w1 = np.random.normal(0, 1, [ind, w])
        # layer 1 bias
        self.b1 = np.random.normal(0, 1, [w])
        # layer 2 weight
        self.w2 = np.random.normal(0, 1, [w, w2])
        # layer 2 bias
        self.b2 = np.random.normal(0, 1, [w2])
        # output layer weight
        self.wout = np.random.normal(0, 1, [w2, outd])
        # output layer bias
        self.bout = np.random.normal(0, 1, [outd])
        # learning rate
        self.lr = lr

    def forward(self, x):
        # input tensor
        self.z1 = x
        # layer 1 output tensor
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # layer 2 output tensor
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # output layer tensor
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        # get gradients for weight and bias
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        # update weight and bias
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        # get gradients for weight and bias
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        # update weight and bias
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
  
        # get gradients for weight and bias
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        # update weight and bias
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

# sigmoid
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train
def train_nn(nn, train_x, train_t, iteration_N=5000):
    for i in range(5000):
        # feed-forward data
        nn.forward(train_x)
        #print("ite>>", i, 'y >>', nn.forward(train_x))
        # update parameters
        nn.train(train_x, train_t)

    return nn


# test
def test_nn(nn, test_x, test_t):
    for j in range(len(test_x)):
        x = train_x[j]
        t = train_t[j]
        print("in:", x, "pred:", nn.forward(x))

# train data
train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)

# train label data
train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

# prepare neural network
nn = NN()

# train
nn = train_nn(nn, train_x, train_t, iteration_N=5000)

# test
test_nn(nn, train_x, train_t)

```

答案：

```bash
in: [0. 0.] pred: [0.03724313]
in: [0. 1.] pred: [0.95885516]
in: [1. 0.] pred: [0.9641076]
in: [1. 1.] pred: [0.03937037]
```

### 96.神经网络（Neural Network）第二步——训练

将问题94中准备的200个训练数据的HOG特征值输入到问题95中的神经网络中进行学习。

对于输出大于 0.5 的打上标注 1，小于 0.5 的打上标注 0，对训练数据计算准确率。训练参数如下：

- $\text{learning rate}=0.01$；
- $\text{epoch}=10000$；
- 将裁剪的图像调整为$32\times32$，并计算 HOG 特征量（HOG 中1个cell的大小为$8\times8$）。

```bash
Accuracy >> 1.0 (200.0 / 200)
```

```Python
import cv2
import numpy as np

np.random.seed(0)

# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
  
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram


# get IoU overlap ratio
def iou(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
    iou_w = iou_x2 - iou_x1
	# get height of IoU
    iou_h = iou_y2 - iou_y1

	# get area of IoU
    area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

# resize using bi-linear
def resize(img, h, w):
    # get shape
    _h, _w, _c  = img.shape

    # get resize ratio
    ah = 1. * h / _h
    aw = 1. * w / _w

    # get index of each y
    y = np.arange(h).repeat(w).reshape(w, -1)
    # get index of each x
    x = np.tile(np.arange(w), (h, 1))

    # get coordinate toward x and y of resized image
    y = (y / ah)
    x = (x / aw)

    # transfer to int
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)

    # clip index
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    # get distance between original image index and resized image index
    dx = x - ix
    dy = y - iy

    dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
    dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)
  
    # resize
    out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
    out[out > 255] = 255

    return out

# neural network
class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        # layer 1 weight
        self.w1 = np.random.normal(0, 1, [ind, w])
        # layer 1 bias
        self.b1 = np.random.normal(0, 1, [w])
        # layer 2 weight
        self.w2 = np.random.normal(0, 1, [w, w2])
        # layer 2 bias
        self.b2 = np.random.normal(0, 1, [w2])
        # output layer weight
        self.wout = np.random.normal(0, 1, [w2, outd])
        # output layer bias
        self.bout = np.random.normal(0, 1, [outd])
        # learning rate
        self.lr = lr

    def forward(self, x):
        # input tensor
        self.z1 = x
        # layer 1 output tensor
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # layer 2 output tensor
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # output layer tensor
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        # get gradients for weight and bias
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        # update weight and bias
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        # get gradients for weight and bias
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        # update weight and bias
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
  
        # get gradients for weight and bias
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        # update weight and bias
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

# sigmoid
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train
def train_nn(nn, train_x, train_t, iteration_N=10000):
    # each iteration
    for i in range(iteration_N):
        # feed-forward data
        nn.forward(train_x)
        # update parameter
        nn.train(train_x, train_t)

    return nn

# test
def test_nn(nn, test_x, test_t, pred_th=0.5):
    accuracy_N = 0.

    # each data
    for data, t in zip(test_x, test_t):
        # get prediction
        prob = nn.forward(data)

        # count accuracy
        pred = 1 if prob >= pred_th else 0
        if t == pred:
            accuracy_N += 1

    # get accuracy 
    accuracy = accuracy_N / len(db)

    print("Accuracy >> {} ({} / {})".format(accuracy, accuracy_N, len(db)))


# crop bounding box and make dataset
def make_dataset(img, gt, Crop_N=200, L=60, th=0.5, H_size=32):
    # get shape
    H, W, _ = img.shape

    # get HOG feature dimension
    HOG_feature_N = ((H_size // 8) ** 2) * 9

    # prepare database
    db = np.zeros([Crop_N, HOG_feature_N + 1])

    # each crop
    for i in range(Crop_N):
        # get left top x of crop bounding box
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L

        # get bounding box
        crop = np.array((x1, y1, x2, y2))

        _iou = np.zeros((3,))
        _iou[0] = iou(gt, crop)
        #_iou[1] = iou(gt2, crop)
        #_iou[2] = iou(gt3, crop)

        # get label
        if _iou.max() >= th:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            label = 1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            label = 0

        # crop area
        crop_area = img[y1:y2, x1:x2]

        # resize crop area
        crop_area = resize(crop_area, H_size, H_size)

        # get HOG feature
        _hog = HOG(crop_area)
  
        # store HOG feature and label
        db[i, :HOG_feature_N] = _hog.ravel()
        db[i, -1] = label

    return db

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG
histogram = HOG(img)

# prepare gt bounding box
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# get database
db = make_dataset(img, gt)


# train neural network
# get input feature dimension
input_dim = db.shape[1] - 1
# prepare train data X
train_x = db[:, :input_dim]
# prepare train data t
train_t = db[:, -1][..., None]

# prepare neural network
nn = NN(ind=input_dim, lr=0.01)
# training
nn = train_nn(nn, train_x, train_t, iteration_N=10000)

# test
test_nn(nn, train_x, train_t)
```

### 97.简单物体检测第一步----滑动窗口（Sliding Window）+HOG

从这里开始进行物体检测吧！

物体检测是检测图像中到底有什么东西的任务。例如，图像在$[x_1, y_1, x_2, y_2]$处有一只狗。像这样把物体圈出来的矩形我们称之为Bounding-box。

下面实现简单物体检测算法：

1. 从图像左上角开始进行滑动窗口扫描；
2. 在滑动的过程中，会依次圈出很多矩形区域；
3. 裁剪出每个矩形区域对应的图像，并对裁剪出的图像提取特征（HOG，SIFT等）；
4. 使用分类器（CNN，SVM等）以确定每个矩形是否包含目标。

这样做的话，会得到一些裁剪过的图像和其对应的矩形的坐标。目前，物体检测主要通过深度学习（Faster R-CNN、YOLO、SSD等）进行，但是这种滑动窗口方法在深度学习开始流行之前已成为主流。为了学习检测的基础知识我们使用滑动窗口来进行检测。

我们实现步骤1至步骤3。

在 `imorimany.jpg`上检测蝾螈的头吧！条件如下：

- 矩形使用以下方法表示：

```python
# [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
```

- 滑动步长为4个像素（每次滑动一个像素固然是好的，但这样需要大量计算，处理时间会变长）；
- 如果矩形超过图像边界，改变矩形的形状使其不超过图像的边界；
- 将裁剪出的矩形部分大小调整为$32\times32$；
- 计算HOG特征值时 cell 大小取$8\times8$。

```Python
import cv2
import numpy as np

np.random.seed(0)

# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
  
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram


# get IoU overlap ratio
def iou(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
    iou_w = iou_x2 - iou_x1
	# get height of IoU
    iou_h = iou_y2 - iou_y1

	# get area of IoU
    area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

# resize using bi-linear
def resize(img, h, w):
    # get shape
    _h, _w, _c  = img.shape

    # get resize ratio
    ah = 1. * h / _h
    aw = 1. * w / _w

    # get index of each y
    y = np.arange(h).repeat(w).reshape(w, -1)
    # get index of each x
    x = np.tile(np.arange(w), (h, 1))

    # get coordinate toward x and y of resized image
    y = (y / ah)
    x = (x / aw)

    # transfer to int
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)

    # clip index
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    # get distance between original image index and resized image index
    dx = x - ix
    dy = y - iy

    dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
    dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)
  
    # resize
    out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
    out[out > 255] = 255

    return out

# sliding window
def sliding_window(img, H_size=32):
    # get shape
    H, W, _ = img.shape
  
    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    # sliding window
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[max(y - dh, 0) : min(y + dh, H), max(x - dw, 0) : min(x + dw, W)]

                # resize crop region
                region = resize(region, H_size, H_size)

                # get HOG feature
                region_hog = HOG(region).ravel()



# read detect target image
img = cv2.imread("imori_many.jpg")

sliding_window(img)
```

### 98.简单物体检测第二步——滑动窗口（Sliding Window）+ NN

对于 `imorimany.jpg`，将问题九十七中求得的各个矩形的HOG特征值输入问题九十六中训练好的神经网络中进行蝾螈头部识别。

在此，绘制$\text{Score}$（即预测是否是蝾螈头部图像的概率）大于$0.7$的矩形。

下面的答案内容为检测矩形的$[x1, y1, x2, y2, \text{Score}]$：

```bash
[[ 27.           0.          69.          21.           0.74268049]
[ 31.           0.          73.          21.           0.89631011]
[ 52.           0.         108.          36.           0.84373157]
[165.           0.         235.          43.           0.73741703]
[ 55.           0.          97.          33.           0.70987278]
[165.           0.         235.          47.           0.92333214]
[169.           0.         239.          47.           0.84030839]
[ 51.           0.          93.          37.           0.84301022]
[168.           0.         224.          44.           0.79237294]
[165.           0.         235.          51.           0.86038564]
[ 51.           0.          93.          41.           0.85151915]
[ 48.           0.         104.          56.           0.73268318]
[168.           0.         224.          56.           0.86675902]
[ 43.          15.          85.          57.           0.93562483]
[ 13.          37.          83.         107.           0.77192307]
[180.          44.         236.         100.           0.82054873]
[173.          37.         243.         107.           0.8478805 ]
[177.          37.         247.         107.           0.87183443]
[ 24.          68.          80.         124.           0.7279032 ]
[103.          75.         145.         117.           0.73725153]
[104.          68.         160.         124.           0.71314282]
[ 96.          72.         152.         128.           0.86269195]
[100.          72.         156.         128.           0.98826957]
[ 25.          69.          95.         139.           0.73449174]
[100.          76.         156.         132.           0.74963093]
[104.          76.         160.         132.           0.96620193]
[ 75.          91.         117.         133.           0.80533424]
[ 97.          77.         167.         144.           0.7852362 ]
[ 97.          81.         167.         144.           0.70371708]]
```

|                                    输入                                     |                                    输出                                    |
| :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_many.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_98.jpg) |

```Python
import cv2
import numpy as np

np.random.seed(0)

# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Magnitude and gradient
    def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

    # get magnitude and gradient
    def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

        return histogram

		# histogram normalization
    def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
    	    for x in range(cell_N_W):
       	    #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
  
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram


# get IoU overlap ratio
def iou(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
    iou_w = iou_x2 - iou_x1
	# get height of IoU
    iou_h = iou_y2 - iou_y1

	# get area of IoU
    area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

# resize using bi-linear
def resize(img, h, w):
    # get shape
    _h, _w, _c  = img.shape

    # get resize ratio
    ah = 1. * h / _h
    aw = 1. * w / _w

    # get index of each y
    y = np.arange(h).repeat(w).reshape(w, -1)
    # get index of each x
    x = np.tile(np.arange(w), (h, 1))

    # get coordinate toward x and y of resized image
    y = (y / ah)
    x = (x / aw)

    # transfer to int
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)

    # clip index
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    # get distance between original image index and resized image index
    dx = x - ix
    dy = y - iy

    dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
    dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)
  
    # resize
    out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
    out[out > 255] = 255

    return out


# neural network
class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        # layer 1 weight
        self.w1 = np.random.normal(0, 1, [ind, w])
        # layer 1 bias
        self.b1 = np.random.normal(0, 1, [w])
        # layer 2 weight
        self.w2 = np.random.normal(0, 1, [w, w2])
        # layer 2 bias
        self.b2 = np.random.normal(0, 1, [w2])
        # output layer weight
        self.wout = np.random.normal(0, 1, [w2, outd])
        # output layer bias
        self.bout = np.random.normal(0, 1, [outd])
        # learning rate
        self.lr = lr

    def forward(self, x):
        # input tensor
        self.z1 = x
        # layer 1 output tensor
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # layer 2 output tensor
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # output layer tensor
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        # get gradients for weight and bias
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        # update weight and bias
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        # get gradients for weight and bias
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        # update weight and bias
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
  
        # get gradients for weight and bias
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        # update weight and bias
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

# sigmoid
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train
def train_nn(nn, train_x, train_t, iteration_N=10000):
    # each iteration
    for i in range(iteration_N):
        # feed-forward data
        nn.forward(train_x)
        # update parameter
        nn.train(train_x, train_t)

    return nn


# crop bounding box and make dataset
def make_dataset(img, gt, Crop_N=200, L=60, th=0.5, H_size=32):
    # get shape
    H, W, _ = img.shape

    # get HOG feature dimension
    HOG_feature_N = ((H_size // 8) ** 2) * 9

    # prepare database
    db = np.zeros([Crop_N, HOG_feature_N + 1])

    # each crop
    for i in range(Crop_N):
        # get left top x of crop bounding box
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L

        # get bounding box
        crop = np.array((x1, y1, x2, y2))

        _iou = np.zeros((3,))
        _iou[0] = iou(gt, crop)
        #_iou[1] = iou(gt2, crop)
        #_iou[2] = iou(gt3, crop)

        # get label
        if _iou.max() >= th:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            label = 1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            label = 0

        # crop area
        crop_area = img[y1:y2, x1:x2]

        # resize crop area
        crop_area = resize(crop_area, H_size, H_size)

        # get HOG feature
        _hog = HOG(crop_area)
  
        # store HOG feature and label
        db[i, :HOG_feature_N] = _hog.ravel()
        db[i, -1] = label

    return db


# sliding window
def sliding_window(img, nn, H_size=32, prob_th=0.7):
    # get shape
    H, W, _ = img.shape

    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    # detected region
    detects = np.ndarray((0, 5), dtype=np.float32)

    # sliding window
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[max(y - dh, 0) : min(y + dh, H), max(x - dw, 0) : min(x + dw, W)]

                # resize crop region
                region = resize(region, H_size, H_size)

                # get HOG feature
                region_hog = HOG(region).ravel()

                # predict score using neural network
                score = nn.forward(region_hog)

                if score >= prob_th:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
                    detects = np.vstack((detects, np.array((x1, y1, x2, y2, score))))

    print(detects)

    return img


# Read image
img = cv2.imread("imori_1.jpg").astype(np.float32)

# prepare gt bounding box
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# get database
db = make_dataset(img, gt)


# train neural network
# get input feature dimension
input_dim = db.shape[1] - 1
# prepare train data X
train_x = db[:, :input_dim]
# prepare train data t
train_t = db[:, -1][..., None]

# prepare neural network
nn = NN(ind=input_dim, w=64, w2=64, lr=0.01)
# training
nn = train_nn(nn, train_x, train_t, iteration_N=10000)


# read detect target image
img2 = cv2.imread("imori_many.jpg")

# detection
out = sliding_window(img2, nn)


cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)

```

### 99.简单物体检测第三步——非极大值抑制（Non-Maximum Suppression）

虽然使用问题九十七中的方法可以粗略地检测出目标，但是 Bounding-box 的数量过多，这对于后面的处理流程是十分不便的。因此，使用非极大值抑制（Non-Maximum Suppression）减少矩形的数量。

NMS是一种留下高分Bounding-box的方法，算法如下：

1. 将Bounding-box的集合$B$按照$\text{Score}$从高到低排序；
2. $\text{Score}$最高的记为$b_0$；
3. 计算$b_0$和其它Bounding-box的$\text{IoU}$。从$B$中删除高于$\text{IoU}$阈值$t$的Bounding-box。将$b_0$添加到输出集合$R$中，并从$B$中删除。
4. 重复步骤2和步骤3直到$B$中没有任何元素；
5. 输出$R$。

在问题九十八的基础上增加NMS（阈值$t=0.25$），并输出图像。请在答案中Bounding-box的左上角附上$\text{Score}$。

不管准确度如何，这样就完成了图像检测的一系列流程。通过增加神经网络，可以进一步提高检测精度。

|                                    输入                                     |                                   NMS前                                    |                        NMS後(answers/answer_99.jpg)                        |
| :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_many.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_98.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_99.jpg) |

```Python
import cv2
import numpy as np

np.random.seed(0)

# read image
img = cv2.imread("imori_1.jpg")
H, W, C = img.shape

# Grayscale
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

gt = np.array((47, 41, 129, 103), dtype=np.float32)

cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,255), 1)

def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])
    iou_w = max(iou_x2 - iou_x1, 0)
    iou_h = max(iou_y2 - iou_y1, 0)
    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)
    return iou


def hog(gray):
    h, w = gray.shape
    # Magnitude and gradient
    gray = np.pad(gray, (1, 1), 'edge')

    gx = gray[1:h+1, 2:] - gray[1:h+1, :w]
    gy = gray[2:, 1:w+1] - gray[:h, 1:w+1]
    gx[gx == 0] = 0.000001

    mag = np.sqrt(gx ** 2 + gy ** 2)
    gra = np.arctan(gy / gx)
    gra[gra<0] = np.pi / 2 + gra[gra < 0] + np.pi / 2

    # Gradient histogram
    gra_n = np.zeros_like(gra, dtype=np.int)

    d = np.pi / 9
    for i in range(9):
        gra_n[np.where((gra >= d * i) & (gra <= d * (i+1)))] = i

    N = 8
    HH = h // N
    HW = w // N
    Hist = np.zeros((HH, HW, 9), dtype=np.float32)
    for y in range(HH):
        for x in range(HW):
            for j in range(N):
                for i in range(N):
                    Hist[y, x, gra_n[y*4+j, x*4+i]] += mag[y*4+j, x*4+i]
          
    ## Normalization
    C = 3
    eps = 1
    for y in range(HH):
        for x in range(HW):
            #for i in range(9):
            Hist[y, x] /= np.sqrt(np.sum(Hist[max(y-1,0):min(y+2, HH), max(x-1,0):min(x+2, HW)] ** 2) + eps)

    return Hist

def resize(img, h, w):
    _h, _w  = img.shape
    ah = 1. * h / _h
    aw = 1. * w / _w
    y = np.arange(h).repeat(w).reshape(w, -1)
    x = np.tile(np.arange(w), (h, 1))
    y = (y / ah)
    x = (x / aw)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    dx = x - ix
    dy = y - iy
  
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    out[out>255] = 255

    return out




# crop and create database

Crop_num = 200
L = 60
H_size = 32
F_n = ((H_size // 8) ** 2) * 9

db = np.zeros((Crop_num, F_n+1))

for i in range(Crop_num):
    x1 = np.random.randint(W-L)
    y1 = np.random.randint(H-L)
    x2 = x1 + L
    y2 = y1 + L
    crop = np.array((x1, y1, x2, y2))

    _iou = iou(gt, crop)

    if _iou >= 0.5:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        label = 1
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        label = 0

    crop_area = gray[y1:y2, x1:x2]
    crop_area = resize(crop_area, H_size, H_size)
    _hog = hog(crop_area)
  
    db[i, :F_n] = _hog.ravel()
    db[i, -1] = label


class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
  
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
  

## training neural network
nn = NN(ind=F_n, lr=0.01)
for i in range(10000):
    nn.forward(db[:, :F_n])
    nn.train(db[:, :F_n], db[:, -1][..., None])


# read detect target image
img2 = cv2.imread("imori_many.jpg")
H2, W2, C2 = img2.shape

# Grayscale
gray2 = 0.2126 * img2[..., 2] + 0.7152 * img2[..., 1] + 0.0722 * img2[..., 0]

# [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

detects = np.ndarray((0, 5), dtype=np.float32)

# sliding window
for y in range(0, H2, 4):
    for x in range(0, W2, 4):
        for rec in recs:
            dh = int(rec[0] // 2)
            dw = int(rec[1] // 2)
            x1 = max(x-dw, 0)
            x2 = min(x+dw, W2)
            y1 = max(y-dh, 0)
            y2 = min(y+dh, H2)
            region = gray2[max(y-dh,0):min(y+dh,H2), max(x-dw,0):min(x+dw,W2)]
            region = resize(region, H_size, H_size)
            region_hog = hog(region).ravel()

            score = nn.forward(region_hog)
            if score >= 0.7:
                #cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,255), 1)
                detects = np.vstack((detects, np.array((x1, y1, x2, y2, score))))


# Non-maximum suppression
def nms(_bboxes, iou_th=0.5, select_num=None, prob_th=None):
    #
    # Non Maximum Suppression
    #
    # Argument
    #  bboxes(Nx5) ... [bbox-num, 5(leftTopX,leftTopY,w,h, score)]
    #  iou_th([float]) ... threshold for iou between bboxes.
    #  select_num([int]) ... max number for choice bboxes. If None, this is unvalid.
    #  prob_th([float]) ... probability threshold to choice. If None, this is unvalid.
    # Return
    #  inds ... choced indices for bboxes
    #

    bboxes = _bboxes.copy()
  
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
  
    # Sort by bbox's score. High -> Low
    sort_inds = np.argsort(bboxes[:, -1])[::-1]

    processed_bbox_ind = []
    return_inds = []

    unselected_inds = sort_inds.copy()
  
    while len(unselected_inds) > 0:
        process_bboxes = bboxes[unselected_inds]
        argmax_score_ind = np.argmax(process_bboxes[::, -1])
        max_score_ind = unselected_inds[argmax_score_ind]
        return_inds += [max_score_ind]
        unselected_inds = np.delete(unselected_inds, argmax_score_ind)

        base_bbox = bboxes[max_score_ind]
        compare_bboxes = bboxes[unselected_inds]
  
        base_x1 = base_bbox[0]
        base_y1 = base_bbox[1]
        base_x2 = base_bbox[2] + base_x1
        base_y2 = base_bbox[3] + base_y1
        base_w = np.maximum(base_bbox[2], 0)
        base_h = np.maximum(base_bbox[3], 0)
        base_area = base_w * base_h

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_x1, compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_y1, compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_x2, compare_bboxes[:, 2] + compare_bboxes[:, 0])
        iou_y2 = np.minimum(base_y2, compare_bboxes[:, 3] + compare_bboxes[:, 1])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h

        compare_w = np.maximum(compare_bboxes[:, 2], 0)
        compare_h = np.maximum(compare_bboxes[:, 3], 0)
        compare_area = compare_w * compare_h

        # bbox's index which iou ratio over threshold is excluded
        all_area = compare_area + base_area - iou_area
        iou_ratio = np.zeros((len(unselected_inds)))
        iou_ratio[all_area < 0.9] = 0.
        _ind = all_area >= 0.9
        iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]
  
        unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

    if prob_th is not None:
        preds = bboxes[return_inds][:, -1]
        return_inds = np.array(return_inds)[np.where(preds >= prob_th)[0]].tolist()
  
    # pick bbox's index by defined number with higher score
    if select_num is not None:
        return_inds = return_inds[:select_num]

    return return_inds


detects = detects[nms(detects, iou_th=0.25)]

for d in detects:
    v = list(map(int, d[:4]))
    cv2.rectangle(img2, (v[0], v[1]), (v[2], v[3]), (0,0,255), 1)
    cv2.putText(img2, "{:.2f}".format(d[-1]), (v[0], v[1]+9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

cv2.imwrite("out.jpg", img2)
cv2.imshow("result", img2)
cv2.waitKey(0)

```

### 100.简单物体检测第四步——评估（Evaluation）：Precision、Recall、F-Score、mAP

我们对检测效果作出评估。

検出はBounding-boxとそのクラスの２つが一致していないと、精度の評価ができない。对于检测效果，我们有Recall、Precision、F-Score、mAP等评价指标。

> 下面是相关术语中日英对照表：
>
> |     中文     |  English  | 日本語 |
> | :-----------: | :-------: | :----: |
> |    准确率    | Accuracy | 正確度 |
> |  精度/查准率  | Precision | 適合率 |
> | 召回率/查全率 |  Recall  | 再現率 |
>
> 我个人认为“查准率&查全率”比“精度&召回率”更准确，所以下面按照“查准率&查全率”翻译。
>
> 另补混淆矩阵（Confusion Matrix）：
>
> <table class="wikitable" align="center" style="text-align:center; border:none; background:transparent;"width="75%" >
>     <tbody>
>         <tr>
>             <td style="border:none;" colspan="2">
>             </td>
>             <td style="background:#eeeebb;" colspan="2"><b>True condition</b>
>             </td>
>         </tr>
>         <tr>
>             <td style="border:none;">
>             </td>
>             <td style="background:#dddddd;"><a href="/wiki/Statistical_population" title="Statistical population">Total
>                     population</a>
>             </td>
>             <td style="background:#ffffcc;">Condition positive
>             </td>
>             <td style="background:#ddddaa;">Condition negative
>             </td>
>             <td style="background:#eeeecc;font-size:90%;"><a href="/wiki/Prevalence" title="Prevalence">Prevalence</a>
>                 <span style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ Condition
>                             positive</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Total
>                             population</span></span></span>
>             </td>
>             <td style="background:#cceecc;border-left:double silver;font-size:90%;" colspan="2"><a
>                     href="/wiki/Accuracy_and_precision" title="Accuracy and precision">Accuracy</a> (ACC) = <span
>                     style="font-size:118%;"><span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ True positive + Σ
>                             True negative</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Total
>                             population</span></span></span>
>             </td>
>         </tr>
>         <tr>
>             <td style="background:#bbeeee;" rowspan="2"><b>Predicted<br>condition</b>
>             </td>
>             <td style="background:#ccffff;">Predicted condition<br>positive
>             </td>
>             <td style="background:#ccffcc;"><span style="color:#006600;"><b><a href="/wiki/True_positive"
>                             class="mw-redirect" title="True positive">True positive</a></b></span>
>             </td>
>             <td style="background:#eedddd;"><span style="color:#cc0000;"><b><a href="/wiki/False_positive"
>                             class="mw-redirect" title="False positive">False positive</a></b>,<br><a
>                         href="/wiki/Type_I_error" class="mw-redirect" title="Type I error">Type I error</a></span>
>             </td>
>             <td style="background:#ccffee;border-top:double silver;font-size:90%;"><a
>                     href="/wiki/Positive_predictive_value" class="mw-redirect"
>                     title="Positive predictive value">Positive predictive value</a> (PPV), <a
>                     href="/wiki/Precision_(information_retrieval)" class="mw-redirect"
>                     title="Precision (information retrieval)">Precision</a> = <span
>                     style="font-size:118%;white-space:nowrap;"><span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ True
>                             positive</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Predicted condition positive</span></span></span>
>             </td>
>             <td style="background:#cceeff;border-top:double silver;font-size:90%;" colspan="2"><a
>                     href="/wiki/False_discovery_rate" title="False discovery rate">False discovery rate</a> (FDR) =
>                 <span style="font-size:118%;white-space:nowrap;"><span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ False
>                             positive</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Predicted condition positive</span></span></span>
>             </td>
>         </tr>
>         <tr>
>             <td style="background:#aadddd;">Predicted condition<br>negative
>             </td>
>             <td style="background:#ffdddd;"><span style="color:#cc0000;"><b><a href="/wiki/False_negative"
>                             class="mw-redirect" title="False negative">False negative</a></b>,<br><a
>                         href="/wiki/Type_II_error" class="mw-redirect" title="Type II error">Type II error</a></span>
>             </td>
>             <td style="background:#bbeebb;"><span style="color:#006600;"><b><a href="/wiki/True_negative"
>                             class="mw-redirect" title="True negative">True negative</a></b></span>
>             </td>
>             <td style="background:#eeddee;border-bottom:double silver;font-size:90%;"><a
>                     href="/wiki/False_omission_rate" class="mw-redirect" title="False omission rate">False omission
>                     rate</a> (FOR) = <span style="font-size:118%;white-space:nowrap;"><span role="math"
>                         class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ False
>                             negative</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Predicted condition negative</span></span></span>
>             </td>
>             <td style="background:#aaddcc;border-bottom:double silver;font-size:90%;" colspan="2"><a
>                     href="/wiki/Negative_predictive_value" class="mw-redirect"
>                     title="Negative predictive value">Negative predictive value</a> (NPV) = <span
>                     style="font-size:118%;white-space:nowrap;"><span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ True
>                             negative</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Predicted condition negative</span></span></span>
>             </td>
>         </tr>
>         <tr style="font-size:90%;">
>             <td style="border:none;vertical-align:bottom;padding:0 2px 0 0;color:#999999;" colspan="2" rowspan="2">
>             </td>
>             <td style="background:#eeffcc;"><a href="/wiki/True_positive_rate" class="mw-redirect"
>                     title="True positive rate">True positive rate</a> (TPR), <a
>                     href="/wiki/Recall_(information_retrieval)" class="mw-redirect"
>                     title="Recall (information retrieval)">Recall</a>, <a href="/wiki/Sensitivity_(tests)"
>                     class="mw-redirect" title="Sensitivity (tests)">Sensitivity</a>, probability of detection,
>                 <a href="/wiki/Statistical_power" class="mw-redirect" title="Statistical power">Power</a> <span
>                     style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ True
>                             positive</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Condition positive</span></span></span>
>             </td>
>             <td style="background:#eeddbb;"><a href="/wiki/False_positive_rate" title="False positive rate">False
>                     positive rate</a> (FPR), <a href="/wiki/Information_retrieval" title="Information retrieval"><span
>                         class="nowrap">Fall-out</span></a>, probability of false alarm <span
>                     style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ False
>                             positive</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Condition negative</span></span></span>
>             </td>
>             <td style="background:#eeeeee;"><a href="/wiki/Positive_likelihood_ratio" class="mw-redirect"
>                     title="Positive likelihood ratio">Positive likelihood ratio</a> <span class="nowrap">(LR+)</span>
>                 <span style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">TPR</span><span
>                             class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">FPR</span></span></span>
>             </td>
>             <td style="background:#dddddd;" rowspan="2"><a href="/wiki/Diagnostic_odds_ratio"
>                     title="Diagnostic odds ratio">Diagnostic odds ratio</a> (DOR) <span
>                     style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">LR+</span><span
>                             class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">LR−</span></span></span>
>             </td>
>             <td style="background:#ddffdd;border-left:double silver;line-height:2;" rowspan="2"><a href="/wiki/F1_score"
>                     title="F1 score">F<sub>1</sub> score</a> = <span style="font-size:118%;white-space:nowrap;">2 ·
>                     <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Precision ·
>                             Recall</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Precision +
>                             Recall</span></span></span>
>             </td>
>         </tr>
>         <tr style="font-size:90%;">
>             <td style="background:#ffeecc;"><a href="/wiki/False_negative_rate" class="mw-redirect"
>                     title="False negative rate">False negative rate</a> (FNR), Miss rate <span
>                     style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ False
>                             negative</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Condition positive</span></span></span>
>             </td>
>             <td style="background:#ddeebb;"><a href="/wiki/Specificity_(tests)" class="mw-redirect"
>                     title="Specificity (tests)">Specificity</a> (SPC), Selectivity, <a href="/wiki/True_negative_rate"
>                     class="mw-redirect" title="True negative rate">True negative rate</a> (TNR) <span
>                     style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">Σ True
>                             negative</span><span class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ Condition negative</span></span></span>
>             </td>
>             <td style="background:#cccccc;"><a href="/wiki/Negative_likelihood_ratio" class="mw-redirect"
>                     title="Negative likelihood ratio">Negative likelihood ratio</a> <span class="nowrap">(LR−)</span>
>                 <span style="font-size:118%;white-space:nowrap;">= <span role="math" class="sfrac nowrap tion"
>                         style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span
>                             class="num" style="display:block; line-height:1em; margin:0 0.1em;">FNR</span><span
>                             class="slash visualhide"></span><span class="den"
>                             style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">TNR</span></span></span>
>             </td>
>         </tr>
>     </tbody>
> </table>
>
> ——gzr

**查全率（Recall）——在多大程度上检测到了正确的矩形？用来表示涵盖了多少正确答案。取值范围为$[0,1]$：**

$$
\text{Recall}=\frac{G'}{G}
$$

其中：

* $G'$——预测输出中阈值大于$t$，Ground-truth中阈值也大于$t$的矩形的数量；
* $G$——Ground-truth中阈值大于$t$的矩形的数量。

**查准率（Precision）——表示结果在多大程度上是正确的。取值范围为$[0,1]$：**

$$
\text{Precision}=\frac{D'}{D}
$$

其中：

* $D‘$——预测输出中阈值大于$t$，Ground-truth中阈值也大于$t$的矩形的数量；
* $D$——预测输出中阈值大于$t$的矩形的数量。

**F-Score——是查全率（Recall）和查准率（Precision）的调和平均。可以表示两者的平均，取值范围为$[0,1]$：**

$$
\text{F-Score}=\frac{2\  \text{Recall}\  \text{Precision}}{\text{Recall}+\text{Precision}}
$$

在文字检测任务中，通常使用Recall、Precision和F-Score等指标进行评估。

**mAP——Mean Average Precision[^3]。在物体检测任务中，常常使用mAP进行效果评价。mAP的计算方法稍稍有点复杂：**

| 1. 判断检测出的矩形与Ground-truth的$\text{IoU}$是否大于阈值$t$，然后创建一个表。 |                      Detect                      | judge |
| :------------------------------------------------------------------------------: | :----------------------------------------------: |
|                                     detect1                                      | 1（当与Ground-truth的$\text{IoU}\geq t$时为$1$） |
|                                     detect2                                      |  0 （当与Ground-truth的$\text{IoU}< t$时为$0$）  |
|                                     detect3                                      |                        1                         |
2. 初始$\text{mAP} = 0$，上表按从上到下的顺序，judge为1的时候，按在此之上被检测出的矩形，计算$\text{Precision}$，并加到$\text{mAP}$中去。
3. 从表的顶部开始按顺序执行步骤2，完成所有操作后，将$\text{mAP}$除以相加的次数。

上面就是求mAP的方法了。对于上面的例子来说：

1. detect1 为$1$，计算$\text{Precision}$。$\text{Precision} = \frac{1}{1} = 1$，$\text{mAP} = 1$；
2. detect2 为$0$，无视；
3. detect3 为$1$，计算$\text{Precision}$。$\text{Precision} = \frac{2}{3} = 0.67$，$\text{mAP} = 1+0.67=1.67$。
4. 由于$\text{mAP}$进行了两次加法，因此$\text{mAP} = 1.67 \div 2 = 0.835$。

令阈值$t=0.5$，计算查全率、查准率、F-Score和mAP吧。

设下面的矩形为Ground-truth：

```python
# [x1, y1, x2, y2]
GT = np.array(((27, 48, 95, 110), (101, 75, 171, 138)), dtype=np.float32)
```

请将与Ground-truth的$\text{IoU}$为$0.5$以上的矩形用红线表示，其他的用蓝线表示。

解答

```bash
Recall >> 1.00 (2.0 / 2)
Precision >> 0.25 (2.0 / 8)
F-Score >>  0.4
mAP >> 0.0625
```

|                            输入 (imori_many.jpg)                            |                         GT(answers/answer_100_gt.jpg)                          |                        输出(answers/answer_100.jpg)                         |
| :-------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/imori_many.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_100_gt.jpg) | ![](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/answer_100.jpg) |

```Python
import cv2
import numpy as np

np.random.seed(0)

# read image
img = cv2.imread("imori_1.jpg")
H, W, C = img.shape

# Grayscale
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

gt = np.array((47, 41, 129, 103), dtype=np.float32)

cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,255), 1)

def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])
    iou_w = max(iou_x2 - iou_x1, 0)
    iou_h = max(iou_y2 - iou_y1, 0)
    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)
    return iou


def hog(gray):
    h, w = gray.shape
    # Magnitude and gradient
    gray = np.pad(gray, (1, 1), 'edge')

    gx = gray[1:h+1, 2:] - gray[1:h+1, :w]
    gy = gray[2:, 1:w+1] - gray[:h, 1:w+1]
    gx[gx == 0] = 0.000001

    mag = np.sqrt(gx ** 2 + gy ** 2)
    gra = np.arctan(gy / gx)
    gra[gra<0] = np.pi / 2 + gra[gra < 0] + np.pi / 2

    # Gradient histogram
    gra_n = np.zeros_like(gra, dtype=np.int)

    d = np.pi / 9
    for i in range(9):
        gra_n[np.where((gra >= d * i) & (gra <= d * (i+1)))] = i

    N = 8
    HH = h // N
    HW = w // N
    Hist = np.zeros((HH, HW, 9), dtype=np.float32)
    for y in range(HH):
        for x in range(HW):
            for j in range(N):
                for i in range(N):
                    Hist[y, x, gra_n[y*4+j, x*4+i]] += mag[y*4+j, x*4+i]
          
    ## Normalization
    C = 3
    eps = 1
    for y in range(HH):
        for x in range(HW):
            #for i in range(9):
            Hist[y, x] /= np.sqrt(np.sum(Hist[max(y-1,0):min(y+2, HH), max(x-1,0):min(x+2, HW)] ** 2) + eps)

    return Hist

def resize(img, h, w):
    _h, _w  = img.shape
    ah = 1. * h / _h
    aw = 1. * w / _w
    y = np.arange(h).repeat(w).reshape(w, -1)
    x = np.tile(np.arange(w), (h, 1))
    y = (y / ah)
    x = (x / aw)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    dx = x - ix
    dy = y - iy
  
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    out[out>255] = 255

    return out

class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
  
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# crop and create database

Crop_num = 200
L = 60
H_size = 32
F_n = ((H_size // 8) ** 2) * 9

db = np.zeros((Crop_num, F_n+1))

for i in range(Crop_num):
    x1 = np.random.randint(W-L)
    y1 = np.random.randint(H-L)
    x2 = x1 + L
    y2 = y1 + L
    crop = np.array((x1, y1, x2, y2))

    _iou = iou(gt, crop)

    if _iou >= 0.5:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        label = 1
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        label = 0

    crop_area = gray[y1:y2, x1:x2]
    crop_area = resize(crop_area, H_size, H_size)
    _hog = hog(crop_area)
  
    db[i, :F_n] = _hog.ravel()
    db[i, -1] = label


## training neural network
nn = NN(ind=F_n, lr=0.01)
for i in range(10000):
    nn.forward(db[:, :F_n])
    nn.train(db[:, :F_n], db[:, -1][..., None])


# read detect target image
img2 = cv2.imread("imori_many.jpg")
H2, W2, C2 = img2.shape

# Grayscale
gray2 = 0.2126 * img2[..., 2] + 0.7152 * img2[..., 1] + 0.0722 * img2[..., 0]

# [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

detects = np.ndarray((0, 5), dtype=np.float32)

# sliding window
for y in range(0, H2, 4):
    for x in range(0, W2, 4):
        for rec in recs:
            dh = int(rec[0] // 2)
            dw = int(rec[1] // 2)
            x1 = max(x-dw, 0)
            x2 = min(x+dw, W2)
            y1 = max(y-dh, 0)
            y2 = min(y+dh, H2)
            region = gray2[max(y-dh,0):min(y+dh,H2), max(x-dw,0):min(x+dw,W2)]
            region = resize(region, H_size, H_size)
            region_hog = hog(region).ravel()

            score = nn.forward(region_hog)
            if score >= 0.7:
                #cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,255), 1)
                detects = np.vstack((detects, np.array((x1, y1, x2, y2, score))))


# Non-maximum suppression
def nms(_bboxes, iou_th=0.5, select_num=None, prob_th=None):
    #
    # Non Maximum Suppression
    #
    # Argument
    #  bboxes(Nx5) ... [bbox-num, 5(leftTopX,leftTopY,w,h, score)]
    #  iou_th([float]) ... threshold for iou between bboxes.
    #  select_num([int]) ... max number for choice bboxes. If None, this is unvalid.
    #  prob_th([float]) ... probability threshold to choice. If None, this is unvalid.
    # Return
    #  inds ... choced indices for bboxes
    #

    bboxes = _bboxes.copy()
  
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
  
    # Sort by bbox's score. High -> Low
    sort_inds = np.argsort(bboxes[:, -1])[::-1]

    processed_bbox_ind = []
    return_inds = []

    unselected_inds = sort_inds.copy()
  
    while len(unselected_inds) > 0:
        process_bboxes = bboxes[unselected_inds]
        argmax_score_ind = np.argmax(process_bboxes[::, -1])
        max_score_ind = unselected_inds[argmax_score_ind]
        return_inds += [max_score_ind]
        unselected_inds = np.delete(unselected_inds, argmax_score_ind)

        base_bbox = bboxes[max_score_ind]
        compare_bboxes = bboxes[unselected_inds]
  
        base_x1 = base_bbox[0]
        base_y1 = base_bbox[1]
        base_x2 = base_bbox[2] + base_x1
        base_y2 = base_bbox[3] + base_y1
        base_w = np.maximum(base_bbox[2], 0)
        base_h = np.maximum(base_bbox[3], 0)
        base_area = base_w * base_h

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_x1, compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_y1, compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_x2, compare_bboxes[:, 2] + compare_bboxes[:, 0])
        iou_y2 = np.minimum(base_y2, compare_bboxes[:, 3] + compare_bboxes[:, 1])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h

        compare_w = np.maximum(compare_bboxes[:, 2], 0)
        compare_h = np.maximum(compare_bboxes[:, 3], 0)
        compare_area = compare_w * compare_h

        # bbox's index which iou ratio over threshold is excluded
        all_area = compare_area + base_area - iou_area
        iou_ratio = np.zeros((len(unselected_inds)))
        iou_ratio[all_area < 0.9] = 0.
        _ind = all_area >= 0.9
        iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]
  
        unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

    if prob_th is not None:
        preds = bboxes[return_inds][:, -1]
        return_inds = np.array(return_inds)[np.where(preds >= prob_th)[0]].tolist()
  
    # pick bbox's index by defined number with higher score
    if select_num is not None:
        return_inds = return_inds[:select_num]

    return return_inds


detects = detects[nms(detects, iou_th=0.25)]


# Evaluation

# [x1, y1, x2, y2]
GT = np.array(((27, 48, 95, 110), (101, 75, 171, 138)), dtype=np.float32)

## Recall, Precision, F-score
iou_th = 0.5

Rs = np.zeros((len(GT)))
Ps = np.zeros((len(detects)))

for i, g in enumerate(GT):
    iou_x1 = np.maximum(g[0], detects[:, 0])
    iou_y1 = np.maximum(g[1], detects[:, 1])
    iou_x2 = np.minimum(g[2], detects[:, 2])
    iou_y2 = np.minimum(g[3], detects[:, 3])
    iou_w = np.maximum(0, iou_x2 - iou_x1)
    iou_h = np.maximum(0, iou_y2 - iou_y1)
    iou_area = iou_w * iou_h
    g_area = (g[2] - g[0]) * (g[3] - g[1])
    d_area = (detects[:, 2] - detects[:, 0]) * (detects[:, 3] - detects[:, 1])
    ious = iou_area / (g_area + d_area - iou_area)
  
    Rs[i] = 1 if len(np.where(ious >= iou_th)[0]) > 0 else 0
    Ps[ious >= iou_th] = 1
  
R = np.sum(Rs) / len(Rs)
P = np.sum(Ps) / len(Ps)
F = (2 * P * R) / (P + R) 

print("Recall >> {:.2f} ({} / {})".format(R, np.sum(Rs), len(Rs)))
print("Precision >> {:.2f} ({} / {})".format(P, np.sum(Ps), len(Ps)))
print("F-score >> ", F)

## mAP
mAP = 0.
for i in range(len(detects)):
    mAP += np.sum(Ps[:i]) / (i + 1) * Ps[i]
mAP /= np.sum(Ps)

print("mAP >>", mAP)

# Display
for i in range(len(detects)):
    v = list(map(int, detects[i, :4]))
    if Ps[i] > 0:
        cv2.rectangle(img2, (v[0], v[1]), (v[2], v[3]), (0,0,255), 1)
    else:
        cv2.rectangle(img2, (v[0], v[1]), (v[2], v[3]), (255,0,0), 1)
    cv2.putText(img2, "{:.2f}".format(detects[i, -1]), (v[0], v[1]+9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

for g in GT:
    cv2.rectangle(img2, (g[0], g[1]), (g[2], g[3]), (0,255,0), 1)

cv2.imwrite("out.jpg", img2)
cv2.imshow("result", img2)
cv2.waitKey(0)
```

### 101.拉普拉斯算子(Laplace)

<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E4%BA%8C%E9%98%B6%E5%B7%AE%E5%88%86.png" style="zoom: 50%;" />

二阶差分模版为：

<img src="https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E4%BA%8C%E9%98%B6%E5%B7%AE%E5%88%86%E6%A8%A1%E6%9D%BF.png" style="zoom:50%;" />

[]: 骤一和步骤二看起来是一样的但是条件4和条件5有小小的差别

[]: [这里](http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/lecture11_detectors_descriptors_cs231a.pdf)。

[]: 文较好的讲解见[这里](https://blog.csdn.net/u014203453/article/details/77598997)。

[^2]: 你们看输出图像左下角黑色的那一块，就是这种没有被”分配“到的情况。

[^1]: 这里应该有个公式的，但是它不知道去哪儿了。

[^2]: 下面图里文字的意思：高周波=高频；低周波=低频；入れ替え=替换。
       ![img](https://raw.githubusercontent.com/CYZYZG/CDN/master/img/%E5%82%85%E9%87%8C%E5%8F%B6%E4%BD%8E%E9%80%9A%E6%BB%A4%E6%B3%A2.png)

[^TODO:]: 这里感觉奇奇怪怪的……

[^0]: 关于这个我没有找到什么中文资料，只有两个差不多的PPT文档。下面的译法参照[这里](http://ftp.gongkong.com/UploadFile/datum/2008-10/2008101416013500001.pdf)。另见冈萨雷斯的《数字图像处理》的2.5.1节。

[^1]: 这一步的数学原理可见[这里](https://blog.csdn.net/lwzkiller/article/details/54633670)。

[^3]: 柱状图是直方图的一种（具体区别请见[这里](https://www.zhihu.com/question/26894953)），原文里都写的是“ヒストグラム“、直方图也是摄影里面的一个术语。这里写直方图感觉有点奇怪，所以对直方图中数据做了各种处理得到的统计图翻译成柱状图。
