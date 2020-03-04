#include "myLayer.hpp"
#include <cassert>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace arma;


void ConvLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	// 获取卷积核的参数
	int tF = param.conv_kernels;
	int tC = inShape[1];
	int tH = param.conv_height;
    int tW = param.conv_width;

	// 初始化w和b，in[1]为w,in[2]为b
	if (!in[1])
	{
		in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*np.sqrt(2/fc_net[L-1])
		if (param.conv_weight_init == "msra")
		{
			(*in[1]) *= std::sqrt(2 / (double)(inShape[1] * inShape[2] * inShape[3]));
		}
		else
		{
			(*in[1]) *= 1e-2;
		}
	}
	if (!in[2])
	{
		in[2].reset(new Blob(tF, 1, 1, 1, TRANDN));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
		(*in[2]) *= 1e-2;
	}
	return;
}

void ConvLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	//1.获取输入Blob尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.获取卷积核尺寸
	int tF = param.conv_kernels;   //卷积核个数（由层名称索引得到）		
	int tH = param.conv_height;   //卷积核高
	int tW = param.conv_width;    //卷积核宽  
	int tP = param.conv_pad;        //padding数
	int tS = param.conv_stride;    //滑动步长
	//3.计算卷积后的尺寸
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + 2 * tP - tH) / tS + 1;    //卷积后图片高度
	int Wo = (Wi + 2 * tP - tW) / tS + 1;  //卷积后图片宽度
	//4.赋值输出Blob尺寸
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
	// 输入数据的通道数要和卷积核的通道数一样
	assert(in[0]->get_C() == in[1]->get_C());

	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
    int Wx = in[0]->get_W();

    // 卷积核个数、高度、宽度
	int F = in[1]->get_N();
	int Hw = in[1]->get_H();
	int Ww = in[1]->get_W();

	// 计算输出数据
	int Ho = (Hx + param.conv_pad * 2 - Hw) / param.conv_stride + 1;
	int Wo = (Wx + param.conv_pad * 2 - Ww) / param.conv_stride + 1;

	// 做padding操作
	Blob padX = in[0]->pad(param.conv_pad);
	// 卷积计算，遍历计算
	out.reset(new Blob(N, F, Ho, Wo));
	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			for (int hh = 0; hh < Ho; ++hh)
			{
				for (int ww = 0; ww < Wo; ++ww)
				{
				    // 使用cube的自带的截取方法
					cube window = padX[n](span(hh*param.conv_stride, hh*param.conv_stride + Hw - 1),
																span(ww*param.conv_stride, ww*param.conv_stride + Ww - 1), 
																span::all);
					//out = Wx+b
					// accu是将所有元素累加，as_scalar是将元素转化为标量，double
					(*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
				}
			}
		}
	}
	return;
}

void ConvLayer::backward(	const shared_ptr<Blob>& din,   //输入梯度
                            const vector<shared_ptr<Blob>>& cache,
                            vector<shared_ptr<Blob>>& grads,
                            const Param& param		)
{
	// 设置输出梯度Blob大小
    grads[0].reset(new Blob(cache[0]->size(), TZEROS));
    grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	// 获取输入梯度的数据
	int Nd = din->get_N();
	int Cd = din->get_C();
	int Hd = din->get_H();
	int Wd = din->get_W();
	// 获取卷积核的参数
	int Hw = param.conv_height;
	int Ww = param.conv_width;
	int stride = param.conv_stride;

	// 对前向传播进行pad的卷积层进行填充
	Blob pad_X = cache[0]->pad(param.conv_pad);
	Blob pad_dX(pad_X.size(),TZEROS);

	// 遍历传过来的梯度
	for (int n = 0; n < Nd; ++n)
	{
		for (int c = 0; c < Cd; ++c)
		{
			for (int hh = 0; hh < Hd; ++hh)
			{
				for (int ww = 0; ww < Wd; ++ww)
				{
					// 截取输入数据
					cube window = pad_X[n](span(hh*stride, hh*stride + Hw - 1),span(ww*stride, ww*stride + Ww - 1),span::all);
					// 计算梯度
					//dX
					pad_dX[n](span(hh*stride, hh*stride + Hw - 1), span(ww*stride, ww*stride + Ww - 1), span::all)   +=   (*din)[n](hh, ww, c) * (*cache[1])[c];
					//dW  --->grads[1]
					(*grads[1])[c] += (*din)[n](hh, ww, c) * window  / Nd;
					//db   --->grads[2]
					(*grads[2])[c](0,0,0) += (*din)[n](hh, ww, c) / Nd;
				}
			}
		}
	}
	// 去除padding部分
	(*grads[0]) = pad_dX.deletePad(param.conv_pad);
	return;
}


void ReluLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	return;
}

void ReluLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
    // 深拷贝
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void ReluLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
	out.reset(new Blob(*in[0]));
	out->maxIn(0);
	int N = out->get_N();
	// 如果有设置上限阈值的话，进行限制
	double th = param.th;
	if(th != 0)
	{
		for (int i = 0; i < N; ++i)
			(*out)[i].transform([th](double e){return e>th ? th : e; });  //ReLU6 (这是一个经验函数)
	}
	return;
}

void ReluLayer::backward(const shared_ptr<Blob>& din,
											const vector<shared_ptr<Blob>>& cache,
											vector<shared_ptr<Blob>>& grads,
											const Param& param)
{
	// relu和pool一样，只有一部分元素进行前传
	grads[0].reset(new Blob(*cache[0]));

	int N = grads[0]->get_N();
	double th = param.th;  //获取上界阈值
	if (th != 0)
	{
		for (int n = 0; n < N; ++n)
		    // 不在范围内的区域是没有梯度的
			(*grads[0])[n].transform([th](double e) {return e > 0 && e < th ? 1 : 0; });
	}
	else
	{
		for (int n = 0; n < N; ++n)
			(*grads[0])[n].transform([](double e) {return e > 0 ? 1 : 0; });
	}

	(*grads[0]) = (*grads[0]) * (*din);
	return;
}

void TanhLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	return;
}

void TanhLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
    // 深拷贝
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void TanhLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
	out.reset(new Blob(*in[0]));
	int N = in[0]->get_N();
	for (int n = 0; n < N; ++n)
	{
		(*out)[n] = (arma::exp((*in[0])[n]) - arma::exp(-(*in[0])[n])) / (arma::exp((*in[0])[n]) + arma::exp(-(*in[0])[n]));
	}
	return;
}

void TanhLayer::backward(const shared_ptr<Blob>& din,
													const vector<shared_ptr<Blob>>& cache,
													vector<shared_ptr<Blob>>& grads,
													const Param& param)
{
	//step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
	grads[0].reset(new Blob(*cache[0]));

	int N = grads[0]->get_N();
	for (int n = 0; n < N; ++n)
	{
		(*grads[0])[n] = (*din)[n] % (1 - arma::square((arma::exp((*cache[0])[n]) - arma::exp(-(*cache[0])[n])) / (arma::exp((*cache[0])[n]) + arma::exp(-(*cache[0])[n]))));
	}
	return;
}


void PoolLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	return;
}

void PoolLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	// 获取输入尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	// 获取池化参数
	int tH = param.pool_height;
	int tW = param.pool_width;
	int tS = param.pool_stride;
	// 计算结果
	int No = Ni;
	int Co = Ci;
	int Ho = (Hi - tH) / tS + 1;
	int Wo = (Wi - tW) / tS + 1;

	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void PoolLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
	int Wx = in[0]->get_W();

	int Hw = param.pool_height;
	int Ww = param.pool_width;

	int Ho = (Hx  - Hw) / param.pool_stride + 1;
	int Wo = (Wx - Ww) / param.pool_stride + 1;

	out.reset(new Blob(N, C, Ho, Wo));

	for (int n = 0; n < N; ++n)
	{
		for (int c = 0; c < C; ++c)
		{
			for (int hh = 0; hh < Ho; ++hh)
			{
				for (int ww = 0; ww < Wo; ++ww)
				{			
					(*out)[n](hh, ww, c) = (*in[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hw - 1),
																			span(ww*param.pool_stride, ww*param.pool_stride + Ww - 1),
																			span(c, c)).max();
				}
			}
		}
	}
	return;
}

void PoolLayer::backward(const shared_ptr<Blob>& din,
											const vector<shared_ptr<Blob>>& cache,
											vector<shared_ptr<Blob>>& grads,
											const Param& param)
{
    // 由于pool之后，梯度只在一些元素上进行传播，所以要计算pool的掩码
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));

	int Nd = din->get_N();
	int Cd = din->get_C();
	int Hd = din->get_H();
	int Wd = din->get_W();

	int Hp = param.pool_height;
	int Wp = param.pool_width;
	int stride = param.pool_stride;

	for (int n = 0; n < Nd; ++n)
	{
		for (int c = 0; c < Cd; ++c)
		{
			for (int hh = 0; hh < Hd; ++hh)
			{
				for (int ww = 0; ww < Wd; ++ww)
				{
					//获取掩码mask
					mat window = (*cache[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
																		span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
																		span(c, c));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window);  //"=="返回的是一个umat类型的矩阵！umat转换为mat
					//计算梯度
					(*grads[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
											span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
											span(c, c))       +=       mask*(*din)[n](hh, ww, c);  //umat  -/-> mat
				}
			}
		}
	}
	return;
}


void FcLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{

	int tF = param.fc_kernels;
	int tC = inShape[1];
	int tH = inShape[2];
	int tW = inShape[3];

	// 初始化，和卷积层一样
	if (!in[1])
	{
        //标准高斯初始化（μ= 0和σ= 1）
		in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
		// np.randn()*np.sqrt(2/fc_net[L-1])
		if (param.fc_weight_init == "msra")
		{
			(*in[1]) *= std::sqrt(2 / (double)(inShape[1] * inShape[2] * inShape[3]));
		}
		else
		{
			(*in[1]) *= 1e-2;
		}	
	}
	if (!in[2])
	{
		in[2].reset(new Blob(tF, 1, 1, 1, TZEROS));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
	}
	return;
}

void FcLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{

	int No = inShape[0];
	int Co = param.fc_kernels;
	int Ho = 1;
	int Wo = 1;

	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void FcLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
    // 全连接操作和卷积操作是相似的，全连接可以看作是一个特殊的卷积操作。
	if (out)
		out.reset();
	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
	int Wx = in[0]->get_W();

	int F = in[1]->get_N();
	int Hw = in[1]->get_H();
    int Ww = in[1]->get_W();
    // 输入的通道数和输入的宽高和全连接的核的参数必须一样大
	assert(in[0]->get_C() == in[1]->get_C());
	assert(Hx == Hw  && Wx == Ww);

	int Ho =  1;
	int Wo =  1;

	out.reset(new Blob(N, F, Ho, Wo));

	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			(*out)[n](0, 0, f) = accu((*in[0])[n] % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
		}
	}
	return;
}

void FcLayer::backward(const shared_ptr<Blob>& din,
										const vector<shared_ptr<Blob>>& cache,
										vector<shared_ptr<Blob>>& grads,
										const Param& param)
{
	//dX,dW,db  -> X,W,b
    grads[0].reset(new Blob(cache[0]->size(),TZEROS));
    grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	int N = grads[0]->get_N();
    int F = grads[1]->get_N();
	assert(F == cache[1]->get_N());

	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			//dX
			(*grads[0])[n] += (*din)[n](0, 0, f) * (*cache[1])[f];
			//dW
			(*grads[1])[f] += (*din)[n](0, 0, f) * (*cache[0])[n] / N;
			//db
			(*grads[2])[f] += (*din)[n](0, 0, f) / N;
		}
	}
	return;
}

void DropoutLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	return;
}

void DropoutLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void DropoutLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
	double drop_rate = param.drop_rate;
	// drop_rate必须为[0,1]
	assert(drop_rate >= 0 && drop_rate <= 1);

	if (mode == "TRAIN")
	{
		// 生成随机的0、1掩码
		shared_ptr<Blob> in_mask(new Blob(in[0]->size(), TRANDU));
		in_mask->convertIn(drop_rate);
		drop_mask.reset(new Blob(*in_mask));
        // 输入特征*掩码
        // 为了保持输入输出的期望不变，除以1-drop_rate
		out.reset((*in[0]) * (*in_mask) / (1-drop_rate));     //rescale:    输出期望值 = （1 - drop_rate）* 原始期望值  / （1 - drop_rate）
	}
	else
	{
	    // 测试阶段直接输出
		out.reset(new Blob(*in[0]));
	}
}

void DropoutLayer::backward(const shared_ptr<Blob>& din,
													const vector<shared_ptr<Blob>>& cache,
													vector<shared_ptr<Blob>>& grads,
													const Param& param)
{
	double drop_rate = param.drop_rate;
	grads[0].reset(new Blob((*din) * (*drop_mask) / (1 - drop_rate)));
}

// bn层分为了两层，bnlayer和scaleLayer
void BNLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
    // bn是沿着通道的归一化，所以不用考虑N
	int C = inShape[1];
	int H = inShape[2];
	int W = inShape[3];

	if (!in[1])
	{
		in[1].reset(new Blob(1, C, H, W, TZEROS));
	}
	if (!in[2])
	{
		in[2].reset(new Blob(1, C, H, W, TZEROS));
	}

	return;
}

void BNLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void BNLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
	if (out)
		out.reset();
		
	out.reset(new Blob(in[0]->size(), TZEROS));

	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int H = in[0]->get_H();
	int W = in[0]->get_W();
	// 训练模式
	if (mode == "TRAIN")
	{
		// 清空变量
		mean_.reset(new cube(1, 1, C, fill::zeros));
		var_.reset(new cube(1, 1, C, fill::zeros));
		std_.reset(new cube(1, 1, C, fill::zeros));
		// 求负均值，sum操作是在每个通道上都会实行一样的操作，0是行，1是列
		for (int i = 0; i < N; ++i)
			(*mean_) += sum(sum((*in[0])[i], 0), 1) / (H*W);    //cube(1, 1, C)
		(*mean_) /= (-N);    //(负)均值-->cube(1, 1, C)
		// 求方差
		for (int i = 0; i < N; ++i)
			(*var_) += square(sum(sum((*in[0])[i], 0), 1) / (H*W) + (*mean_));
		(*var_) /= N;		//方差-->cube(1, 1, C)
		// 求标准差，为了防止分母为0，加上1e-5
		(*std_) = sqrt((*var_) + 1e-5);   //标准差-->cube(1, 1, C)
		// 广播均值和标准差，完成尺度匹配
		cube mean_tmp(H, W, C, fill::zeros);
		cube std_tmp(H, W, C, fill::zeros);
		for (int c = 0; c < C; ++c)
		{
			mean_tmp.slice(c).fill(as_scalar((*mean_).slice(c)));        //负均值-->cube(H, W, C)
			std_tmp.slice(c).fill(as_scalar((*std_).slice(c)));			//标准差-->cube(H, W, C)
		}
		// 归一化
		for (int i = 0; i < N; ++i)
			(*out)[i] = ((*in[0])[i] + mean_tmp) / std_tmp;
		// 初始化测试时使用的参数
		if (!running_mean_std_init)
		{
			(*in[1])[0] = mean_tmp;
			(*in[2])[0] = std_tmp;
			running_mean_std_init = true;
		}
		// 移动加权平均，计算整个数据集的方差和均值
		double yita = 0.99;
		(*in[1])[0] = yita*(*in[1])[0] + (1 - yita)*mean_tmp;
		(*in[2])[0] = yita*(*in[2])[0] + (1 - yita)*std_tmp;

	}
	else
	{
		//测试阶段，用 running_mean_和  running__std_来归一化每一个特征
		for (int n = 0; n < N; ++n)
			(*out)[n] = ((*in[0])[n] + (*in[1])[0]) / (*in[2])[0];
	}

	return;
}

void BNLayer::backward(const shared_ptr<Blob>& din,
											const vector<shared_ptr<Blob>>& cache,
											vector<shared_ptr<Blob>>& grads,
											const Param& param)
{
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));//dx  
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	//广播均值和标准差，完成尺寸匹配
	cube mean_tmp(H, W, C, fill::zeros);
	cube var_tmp(H, W, C, fill::zeros);
	cube std_tmp(H, W, C, fill::zeros);
	for (int c = 0; c < C; ++c)
	{
		mean_tmp.slice(c).fill(as_scalar((*mean_).slice(c)));        //负均值-->cube(H, W, C)
		var_tmp.slice(c).fill(as_scalar((*var_).slice(c)));
		std_tmp.slice(c).fill(as_scalar((*std_).slice(c)));			//标准差-->cube(H, W, C)
	}
	//注意：反向传播的计算是可以通过化简来减少计算量的，这里没做化简，直接按原始公式实现！！！
	for (int k = 0; k < N; ++k)
	{
		cube item1(H, W, C, fill::zeros);
		for (int i = 0; i < N; ++i)
			item1 += (*din)[i] % ((*cache[0])[i] + mean_tmp);   //  cube(H, W, C)		
		cube tmp = (-sum(sum(item1, 0), 1) / (2 * (*var_) % (*std_))) / N;    //  cube(1, 1, C)

		cube item2(1, 1, C, fill::zeros);
		for (int j = 0; j < N; ++j)
			item2 += (tmp % (2 * (sum(sum((*cache[0])[j], 0), 1) / (H*W) + (*mean_))));  //  cube(1, 1, C)

		cube item3(H, W, C, fill::zeros);
		for (int i = 0; i < N; ++i)
			item3 += (*din)[i] / std_tmp;    //  cube(H, W, C)

		cube item4(1, 1, C, fill::zeros);
		item4 = sum(sum(item3, 0), 1);  //  cube(1, 1, C)

		//4.广播，完成尺寸匹配
		cube black0 = (item2 + item4) / (-N);   //黑色梯度流
		cube red0 = (tmp % (2 * (sum(sum((*cache[0])[k], 0), 1) / (H*W) + (*mean_))));   //红色梯度流
		cube black_(H, W, C, fill::zeros);  //广播后的黑色梯度流
		cube red_(H, W, C, fill::zeros);	 //广播后的红色梯度流
		cube purple_ = (*din)[k] / std_tmp;
		for (int c = 0; c < C; ++c)
		{
			black_.slice(c).fill(as_scalar(black0.slice(c)));        //cube(H, W, C)
			red_.slice(c).fill(as_scalar(red0.slice(c)));			//cube(H, W, C)
		}	
		(*grads[0])[k] = (black_ + red_) / (H*W) + purple_;
	}
	return;
}

void ScaleLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	int C = inShape[1];

	// 初始化w和b，w和b都是每个通道一个的
	if (!in[1])
	{
		in[1].reset(new Blob(1, C, 1, 1, TONES));
		cout << "initLayer: " << lname << "  Init  γ  with Ones ;" << endl;
	}
	if (!in[2])
	{
		in[2].reset(new Blob(1, C, 1, 1, TZEROS));
		cout << "initLayer: " << lname << "  Init  β  with Zeros ;" << endl;
	}
	return;
}

void ScaleLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void ScaleLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode)
{
    // 清空输出
	out.reset(new Blob(in[0]->size(), TZEROS));

	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int H = in[0]->get_H();
	int W = in[0]->get_W();

	// 通过广播来实现参数相应的匹配
	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> beta(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c)
	{
		(*gamma)[0].slice(c).fill(as_scalar((*in[1])[0].slice(c)));
		(*beta)[0].slice(c).fill(as_scalar((*in[2])[0].slice(c)));
	}
	// 进行平移和缩放
	for (int n = 0; n < N; ++n)
		(*out)[n] = (*gamma)[0] % (*in[0])[n] + (*beta)[0];  //out  = γ * in    +  β

	return;
}

void ScaleLayer::backward(const shared_ptr<Blob>& din,
												const vector<shared_ptr<Blob>>& cache,
												vector<shared_ptr<Blob>>& grads,
												const Param& param)
{
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));//dx  
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));//dγ
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));//dβ
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	//1. γ 通过广播完成尺寸匹配
	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c)
		(*gamma)[0].slice(c).fill(as_scalar((*cache[1])[0].slice(c)));  //因为dx  = din % γ  ，所以γ 需要广播完成尺寸匹配


	//2. 反向计算梯度
	shared_ptr<Blob> dgamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> dbeta(new Blob(1, C, H, W, TZEROS));
	for (int n = 0; n < N; ++n)
	{
		(*grads[0])[n] = (*din)[n] % (*gamma)[0];			 // dx  = din % γ        （N,C,H,W）
		(*dgamma)[0] += (*din)[n] % (*cache[0])[n];    // dγ  = din % x        （1, C,H,W）   注意这边的不同样本的梯度累加
		(*dbeta)[0] += (*din)[n];									 // dβ  = din               （1, C,H,W）   注意这边的不同样本的梯度累加
	}
	(*grads[1])[0] = sum(sum((*dgamma)[0], 0), 1) / N;    //梯度合流后平均dγ
	(*grads[2])[0] = sum(sum((*dbeta)[0], 0), 1) / N;         //梯度合流后平均dβ

	return;
}


void SoftmaxLossLayer::softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
	if (dout)
		dout.reset();
	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
	int Wx = in[0]->get_W();
	assert(Hx == 1 && Wx == 1);
	dout.reset(new Blob(N, C, Hx, Wx));   //（N,C,1,1）
	double loss_ = 0;
	for (int i = 0; i < N; ++i)
	{
	    // softmax
		cube prob = arma::exp((*in[0])[i]) / arma::accu(arma::exp((*in[0])[i]));
		// 交叉熵累加
		loss_ += (-arma::accu((*in[1])[i] % arma::log(prob )));
		// 梯度计算
		(*dout)[i] = prob - (*in[1])[i];
	}
	loss = loss_ / N;   //求平均损失
	return;
}

void SVMLossLayer::hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
	if (dout)
		dout.reset();
	//-------step1.获取相关尺寸
	int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
	int C = in[0]->get_C();         //输入Blob通道数
	int Hx = in[0]->get_H();      //输入Blob高
	int Wx = in[0]->get_W();    //输入Blob宽
	assert(Hx == 1 && Wx == 1);
	dout.reset(new Blob(N, C, Hx, Wx));   //（N,C,1,1）
	double loss_ = 0;
	double delta = 0.2;
	for (int i = 0; i < N; ++i)
	{
		//(1).计算损失
		int idx_max = (*in[1])[i].index_max();
		double positive_x = (*in[0])[i](0, 0, idx_max);
		cube tmp = ((*in[0])[i] - positive_x + delta);           //代入hinge loss公式
		tmp(0, 0, idx_max) = 0;                                              //剔除正确类里面的值
		tmp.transform([](double e) {return e > 0 ? e : 0; });  //做max()操作，得到各个分类的损失
		loss_ +=arma::accu(tmp);  //得到所有类别的损失和

		//(2).计算梯度
		tmp.transform([](double e) {return e ? 1 : 0; });
		tmp(0,0,idx_max)= -arma::accu(tmp);
		(*dout)[i]=tmp;

	}
	loss = loss_ / N;   //求平均损失
	return;
}