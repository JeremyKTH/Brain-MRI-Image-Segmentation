import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def normalize(x, dim=None):
	nrm = (x**2).sum(dim=dim, keepdim=True)**0.5
	nrm[nrm == 0] = 1.
	return x / nrm


class HebbConv2d(nn.Module):
	"""
	A 2d convolutional layer that learns through Hebbian plasticity
	"""
	
	MODE_SWTA = 'swta'
	MODE_HPCA = 'hpca'
	MODE_CONTRASTIVE = 'contrastive'

	def __init__(self, out_channels, in_channels, kernel_size, stride,
	             w_nrm=True, act=nn.ReLU(inplace=True),
	             mode=MODE_SWTA, k=0.02, patchwise=True,
	             contrast=1., uniformity=False):
		"""

		:param out_channels: output channels of the convolutional kernel
		:param in_channels: input channels of the convolutional kernel
		:param kernel_size: size of the convolutional kernel (int or tuple)
		:param stride: stride of the convolutional kernel (int or tuple
		:param w_nrm: whether to normalize the weight vectors before computing outputs
		:param act: the nonlinear activation function after convolution
		:param mode: the learning mode, either 'swta' or 'hpca'
		:param k: softmax inverse temperature parameter used for swta-type learning
		:param patchwise: whether updates for each convolutional patch should be computed separately,
		and then aggregated
		:param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
		:param uniformity: whether to use uniformity weighting in contrastive-type learning.
		
		"""
		
		super().__init__()
		self.mode = mode
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.kernel_size = _pair(kernel_size)
		self.stride = stride

		self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
		nn.init.xavier_normal_(self.weight)
		self.w_nrm = w_nrm
		self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=True)
		self.act = act
		self.register_buffer('delta_w', torch.empty_like(self.weight))

		self.k = k
		self.patchwise = patchwise
		self.contrast = contrast
		self.uniformity = uniformity

	def apply_weights(self, x, w):
		"""
		This function provides the logic for combining input x and weight w
		"""
		
		return torch.conv2d(x, w, self.bias, self.stride)
	
	def forward(self, x):
		w = self.weight
		if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
		y = self.act(self.apply_weights(x, w))
		if self.training: self.update(x, y)
		return y

	def update(self, x, y):
		"""
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_CONTRASTIVE]:
			raise NotImplementedError("Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))
		
		if self.mode == self.MODE_SWTA:
			# Logic for swta-type learning
			x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
			x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
			if self.patchwise:
				r = (y * self.k).softmax(dim=1).permute(1, 0, 2, 3).reshape(self.out_channels, -1)
				dec = r.sum(1, keepdim=True) * self.weight.reshape(self.out_channels, -1)
				self.delta_w[:, :, :, :] = (r.matmul(x_unf) - dec).reshape_as(self.weight)
			else:
				r = (y * self.k).softmax(dim=1).permute(2, 3, 1, 0).reshape(y.size(2), y.size(3), self.out_channels, -1)
				krn = torch.eye(len(self.weight[0]), device=x.device, dtype=x.dtype).view(len(self.weight[0]), self.in_channels, *self.kernel_size)
				dec = torch.conv_transpose2d((r.sum(dim=-1, keepdim=True) * self.weight.reshape(1, 1, self.out_channels, -1)).permute(2, 3, 0, 1), krn, self.stride)
				self.delta_w[:, :, :, :] = (r.permute(2, 3, 0, 1).reshape(self.out_channels, -1).matmul(x_unf) - F.unfold(dec, kernel_size=self.kernel_size, stride=self.stride).sum(dim=-1)).reshape_as(self.weight)
		
		if self.mode == self.MODE_HPCA:
			# Logic for hpca-type learning
			x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
			x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
			if self.patchwise:
				r = y.permute(1, 0, 2, 3).reshape(self.out_channels, -1)
				l = (torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(self.out_channels, 1) <= torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
				dec = (r.matmul(r.transpose(-2, -1)) * l).matmul(self.weight.view(self.out_channels, -1))
				self.delta_w[:, :, :, :] = (r.matmul(x_unf) - dec).reshape_as(self.weight)
			else:
				r = y.permute(2, 3, 1, 0).reshape(y.size(2), y.size(3), self.out_channels, -1)
				l = (torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(self.out_channels, 1) <= torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
				dec = torch.conv_transpose2d((r.matmul(r.transpose(-2, -1)) * l.unsqueeze(0).unsqueeze(1)).permute(3, 2, 0, 1), self.weight, self.stride)
				self.delta_w[:, :, :, :] = (r.permute(2, 3, 0, 1).reshape(self.out_channels, -1).matmul(x_unf) - F.unfold(dec, kernel_size=self.kernel_size, stride=self.stride).sum(dim=-1)).reshape_as(self.weight)
		
		if self.mode == self.MODE_CONTRASTIVE:
			y = normalize(y, dim=1)
			y_unf = F.unfold(y, _pair(3), padding=_pair(1))
			y_unf = y_unf.permute(0, 2, 1).reshape(y_unf.size(0), y_unf.size(2), y.size(1), 9)
			
			# Positive contribution
			L = - y_unf.sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
			
			# Uniformity weighting, if required
			if self.uniformity:
				x = normalize(x, dim=1)
				x_unf = F.unfold(x, _pair(3), padding=_pair(1))
				x_unf = x_unf.permute(0, 2, 1).reshape(x_unf.size(0), x_unf.size(2), x.size(1), 9)
				uniformity_map = (x_unf.sum(-1).reshape(-1, x.size(1)) * x.permute(0, 2, 3, 1).reshape(-1, x.size(1))).sum(dim=-1, keepdim=True)
				uniformity_map = self.apply_weights(uniformity_map.reshape(x.size(0), 1, x.size(1), x.size(2)), torch.ones([1, 1, *self.kernel_size], device=x.device, dtype=x.dtype)).reshape(-1, 1)
				L *= uniformity_map
			
			# Negative contribution
			idx = torch.randperm(y_unf.size(0), device=y_unf.device)
			L += self.contrast * y_unf[idx].sum(-1).reshape(-1, y.size(1)) * y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
			
			# Compute update
			L = L.sum()
			self.zero_grad()
			L.backward()
			self.delta_w[:, :, :, :] = self.weight.grad.clone().detach()
			self.zero_grad()
		
	def local_update(self, alpha=1):
		"""
		This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
		self.weight.grad of the weigth parameter.
		
		This function should be called before optimizer.step(), so that the optimizer will use the locally computed
		update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
		function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
		self.weight.grad, and this function combines this value with self.delta_w as
		self.weight.grad = self.weight.grad - alpha * self.delta_w
		Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.
		
		"""
		
		if self.weight.grad is None: self.weight.grad = -alpha * self.delta_w
		else: self.weight.grad -= alpha * self.delta_w
		self.delta_w.zero_()


class HebbConvTranspose2d(HebbConv2d):
	"""
	A 2d convolutional layer that learns through Hebbian plasticity
	"""
	
	MODE_SWTA_T = 'swta_t'
	MODE_HPCA_T = 'hpca_t'

	def __init__(self, in_channels, out_channels, kernel_size, stride, w_nrm=True, act=nn.ReLU(inplace=True),
	             mode=MODE_SWTA_T, k=0.02, patchwise=False, contrast=1., uniformity=False):
		"""
		
		:param out_channels: output channels of the convolutional kernel
		:param in_channels: input channels of the convolutional kernel
		:param kernel_size: size of the convolutional kernel (int or tuple)
		:param stride: stride of the convolutional kernel (int or tuple
		:param w_nrm: whether to normalize the weight vectors before computing outputs
		:param act: the nonlinear activation function after convolution
		:param mode: the learning mode, either 'swta', 'swta_t, 'hpca', 'hpca_t', or 'contrastive'
		:param k: softmax inverse temperature parameter used for swta-type learning
		:param patchwise: whether updates for each convolutional patch should be computed separately,
		and then aggregated
		:param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
		:param uniformity: whether to use uniformity weighting in contrastive-type learning.
		"""
		
		super().__init__(out_channels, in_channels, kernel_size, stride, w_nrm, act, mode, k, patchwise, contrast, uniformity)

	def apply_weights(self, x, w):
		"""
		This function implements the logic that computes Hebbian plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		return torch.conv_transpose2d(x, w, self.bias, self.stride)

	def update(self, x, y):
		"""
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
		
		if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_SWTA_T, self.MODE_HPCA_T, self.MODE_CONTRASTIVE]:
			raise NotImplementedError("Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))
		
		if self.mode in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_CONTRASTIVE]:
			# In case of swta-type or hpca-type learning, use the learning rules for ordinary convolution,
			# but exchanging x and y
			super().update(y, x)
		
		if self.mode == self.MODE_SWTA_T:
			# Logic for swta-type learning in transpose convolutional layers
			r = (y * self.k).softmax(dim=1)
			r = F.unfold(r, kernel_size=self.kernel_size, stride=self.stride)
			r = r.permute(0, 2, 1).reshape(-1, self.out_channels, self.kernel_size[0]*self.kernel_size[1]).permute(2, 1, 0)
			dec = r.sum(2, keepdim=True) * self.weight.permute(2, 3, 1, 0).reshape(-1, self.out_channels, self.in_channels)
			if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
			self.delta_w[:, :, :, :] = (r.matmul(x.permute(0, 2, 3, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1, 0).reshape_as(self.weight)
		
		if self.mode == self.MODE_HPCA_T:
			# Logic for hpca-type learning in transpose convolutional layers
			r = y
			r = F.unfold(r, kernel_size=self.kernel_size, stride=self.stride)
			r = r.permute(0, 2, 1).reshape(-1, self.out_channels, self.kernel_size[0]*self.kernel_size[1]).permute(2, 1, 0)
			l = (torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(self.out_channels, 1) <= torch.arange(self.out_channels, device=x.device, dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
			dec = (r.matmul(r.transpose(-2, -1)) * l.unsqueeze(0)).matmul(self.weight.permute(2, 3, 1, 0).reshape(-1, self.out_channels, self.in_channels))
			if self.patchwise: dec = dec.sum(dim=0, keepdim=True)
			self.delta_w[:, :, :, :] = (r.matmul(x.permute(0, 2, 3, 1).reshape(1, -1, x.size(1))) - dec).permute(2, 1, 0).reshape_as(self.weight)
		
		