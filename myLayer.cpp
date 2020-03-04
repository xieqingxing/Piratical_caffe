#include "myLayer.hpp"
#include <cassert>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace arma;


void ConvLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	// ��ȡ����˵Ĳ���
	int tF = param.conv_kernels;
	int tC = inShape[1];
	int tH = param.conv_height;
    int tW = param.conv_width;

	// ��ʼ��w��b��in[1]Ϊw,in[2]Ϊb
	if (!in[1])
	{
		in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));  //��׼��˹��ʼ������= 0�ͦ�= 1��    //np.randn()*np.sqrt(2/fc_net[L-1])
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
		in[2].reset(new Blob(tF, 1, 1, 1, TRANDN));  //��׼��˹��ʼ������= 0�ͦ�= 1��    //np.randn()*0.01
		(*in[2]) *= 1e-2;
	}
	return;
}

void ConvLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
	//1.��ȡ����Blob�ߴ�
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.��ȡ����˳ߴ�
	int tF = param.conv_kernels;   //����˸������ɲ����������õ���		
	int tH = param.conv_height;   //����˸�
	int tW = param.conv_width;    //����˿�  
	int tP = param.conv_pad;        //padding��
	int tS = param.conv_stride;    //��������
	//3.��������ĳߴ�
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + 2 * tP - tH) / tS + 1;    //�����ͼƬ�߶�
	int Wo = (Wi + 2 * tP - tW) / tS + 1;  //�����ͼƬ���
	//4.��ֵ���Blob�ߴ�
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
	// �������ݵ�ͨ����Ҫ�;���˵�ͨ����һ��
	assert(in[0]->get_C() == in[1]->get_C());

	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
    int Wx = in[0]->get_W();

    // ����˸������߶ȡ����
	int F = in[1]->get_N();
	int Hw = in[1]->get_H();
	int Ww = in[1]->get_W();

	// �����������
	int Ho = (Hx + param.conv_pad * 2 - Hw) / param.conv_stride + 1;
	int Wo = (Wx + param.conv_pad * 2 - Ww) / param.conv_stride + 1;

	// ��padding����
	Blob padX = in[0]->pad(param.conv_pad);
	// ������㣬��������
	out.reset(new Blob(N, F, Ho, Wo));
	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			for (int hh = 0; hh < Ho; ++hh)
			{
				for (int ww = 0; ww < Wo; ++ww)
				{
				    // ʹ��cube���Դ��Ľ�ȡ����
					cube window = padX[n](span(hh*param.conv_stride, hh*param.conv_stride + Hw - 1),
																span(ww*param.conv_stride, ww*param.conv_stride + Ww - 1), 
																span::all);
					//out = Wx+b
					// accu�ǽ�����Ԫ���ۼӣ�as_scalar�ǽ�Ԫ��ת��Ϊ������double
					(*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
				}
			}
		}
	}
	return;
}

void ConvLayer::backward(	const shared_ptr<Blob>& din,   //�����ݶ�
                            const vector<shared_ptr<Blob>>& cache,
                            vector<shared_ptr<Blob>>& grads,
                            const Param& param		)
{
	// ��������ݶ�Blob��С
    grads[0].reset(new Blob(cache[0]->size(), TZEROS));
    grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	// ��ȡ�����ݶȵ�����
	int Nd = din->get_N();
	int Cd = din->get_C();
	int Hd = din->get_H();
	int Wd = din->get_W();
	// ��ȡ����˵Ĳ���
	int Hw = param.conv_height;
	int Ww = param.conv_width;
	int stride = param.conv_stride;

	// ��ǰ�򴫲�����pad�ľ����������
	Blob pad_X = cache[0]->pad(param.conv_pad);
	Blob pad_dX(pad_X.size(),TZEROS);

	// �������������ݶ�
	for (int n = 0; n < Nd; ++n)
	{
		for (int c = 0; c < Cd; ++c)
		{
			for (int hh = 0; hh < Hd; ++hh)
			{
				for (int ww = 0; ww < Wd; ++ww)
				{
					// ��ȡ��������
					cube window = pad_X[n](span(hh*stride, hh*stride + Hw - 1),span(ww*stride, ww*stride + Ww - 1),span::all);
					// �����ݶ�
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
	// ȥ��padding����
	(*grads[0]) = pad_dX.deletePad(param.conv_pad);
	return;
}


void ReluLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	return;
}

void ReluLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param)
{
    // ���
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
	// ���������������ֵ�Ļ�����������
	double th = param.th;
	if(th != 0)
	{
		for (int i = 0; i < N; ++i)
			(*out)[i].transform([th](double e){return e>th ? th : e; });  //ReLU6 (����һ�����麯��)
	}
	return;
}

void ReluLayer::backward(const shared_ptr<Blob>& din,
											const vector<shared_ptr<Blob>>& cache,
											vector<shared_ptr<Blob>>& grads,
											const Param& param)
{
	// relu��poolһ����ֻ��һ����Ԫ�ؽ���ǰ��
	grads[0].reset(new Blob(*cache[0]));

	int N = grads[0]->get_N();
	double th = param.th;  //��ȡ�Ͻ���ֵ
	if (th != 0)
	{
		for (int n = 0; n < N; ++n)
		    // ���ڷ�Χ�ڵ�������û���ݶȵ�
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
    // ���
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
	//step1. ��������ݶ�Blob�ĳߴ磨dX---grads[0]��
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
	// ��ȡ����ߴ�
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	// ��ȡ�ػ�����
	int tH = param.pool_height;
	int tW = param.pool_width;
	int tS = param.pool_stride;
	// ������
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
    // ����pool֮���ݶ�ֻ��һЩԪ���Ͻ��д���������Ҫ����pool������
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
					//��ȡ����mask
					mat window = (*cache[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
																		span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
																		span(c, c));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window);  //"=="���ص���һ��umat���͵ľ���umatת��Ϊmat
					//�����ݶ�
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

	// ��ʼ�����;����һ��
	if (!in[1])
	{
        //��׼��˹��ʼ������= 0�ͦ�= 1��
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
		in[2].reset(new Blob(tF, 1, 1, 1, TZEROS));  //��׼��˹��ʼ������= 0�ͦ�= 1��    //np.randn()*0.01
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
    // ȫ���Ӳ����;�����������Ƶģ�ȫ���ӿ��Կ�����һ������ľ��������
	if (out)
		out.reset();
	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int Hx = in[0]->get_H();
	int Wx = in[0]->get_W();

	int F = in[1]->get_N();
	int Hw = in[1]->get_H();
    int Ww = in[1]->get_W();
    // �����ͨ����������Ŀ�ߺ�ȫ���ӵĺ˵Ĳ�������һ����
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
	// drop_rate����Ϊ[0,1]
	assert(drop_rate >= 0 && drop_rate <= 1);

	if (mode == "TRAIN")
	{
		// ���������0��1����
		shared_ptr<Blob> in_mask(new Blob(in[0]->size(), TRANDU));
		in_mask->convertIn(drop_rate);
		drop_mask.reset(new Blob(*in_mask));
        // ��������*����
        // Ϊ�˱�������������������䣬����1-drop_rate
		out.reset((*in[0]) * (*in_mask) / (1-drop_rate));     //rescale:    �������ֵ = ��1 - drop_rate��* ԭʼ����ֵ  / ��1 - drop_rate��
	}
	else
	{
	    // ���Խ׶�ֱ�����
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

// bn���Ϊ�����㣬bnlayer��scaleLayer
void BNLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
    // bn������ͨ���Ĺ�һ�������Բ��ÿ���N
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
	// ѵ��ģʽ
	if (mode == "TRAIN")
	{
		// ��ձ���
		mean_.reset(new cube(1, 1, C, fill::zeros));
		var_.reset(new cube(1, 1, C, fill::zeros));
		std_.reset(new cube(1, 1, C, fill::zeros));
		// �󸺾�ֵ��sum��������ÿ��ͨ���϶���ʵ��һ���Ĳ�����0���У�1����
		for (int i = 0; i < N; ++i)
			(*mean_) += sum(sum((*in[0])[i], 0), 1) / (H*W);    //cube(1, 1, C)
		(*mean_) /= (-N);    //(��)��ֵ-->cube(1, 1, C)
		// �󷽲�
		for (int i = 0; i < N; ++i)
			(*var_) += square(sum(sum((*in[0])[i], 0), 1) / (H*W) + (*mean_));
		(*var_) /= N;		//����-->cube(1, 1, C)
		// ���׼�Ϊ�˷�ֹ��ĸΪ0������1e-5
		(*std_) = sqrt((*var_) + 1e-5);   //��׼��-->cube(1, 1, C)
		// �㲥��ֵ�ͱ�׼���ɳ߶�ƥ��
		cube mean_tmp(H, W, C, fill::zeros);
		cube std_tmp(H, W, C, fill::zeros);
		for (int c = 0; c < C; ++c)
		{
			mean_tmp.slice(c).fill(as_scalar((*mean_).slice(c)));        //����ֵ-->cube(H, W, C)
			std_tmp.slice(c).fill(as_scalar((*std_).slice(c)));			//��׼��-->cube(H, W, C)
		}
		// ��һ��
		for (int i = 0; i < N; ++i)
			(*out)[i] = ((*in[0])[i] + mean_tmp) / std_tmp;
		// ��ʼ������ʱʹ�õĲ���
		if (!running_mean_std_init)
		{
			(*in[1])[0] = mean_tmp;
			(*in[2])[0] = std_tmp;
			running_mean_std_init = true;
		}
		// �ƶ���Ȩƽ���������������ݼ��ķ���;�ֵ
		double yita = 0.99;
		(*in[1])[0] = yita*(*in[1])[0] + (1 - yita)*mean_tmp;
		(*in[2])[0] = yita*(*in[2])[0] + (1 - yita)*std_tmp;

	}
	else
	{
		//���Խ׶Σ��� running_mean_��  running__std_����һ��ÿһ������
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

	//�㲥��ֵ�ͱ�׼���ɳߴ�ƥ��
	cube mean_tmp(H, W, C, fill::zeros);
	cube var_tmp(H, W, C, fill::zeros);
	cube std_tmp(H, W, C, fill::zeros);
	for (int c = 0; c < C; ++c)
	{
		mean_tmp.slice(c).fill(as_scalar((*mean_).slice(c)));        //����ֵ-->cube(H, W, C)
		var_tmp.slice(c).fill(as_scalar((*var_).slice(c)));
		std_tmp.slice(c).fill(as_scalar((*std_).slice(c)));			//��׼��-->cube(H, W, C)
	}
	//ע�⣺���򴫲��ļ����ǿ���ͨ�����������ټ������ģ�����û������ֱ�Ӱ�ԭʼ��ʽʵ�֣�����
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

		//4.�㲥����ɳߴ�ƥ��
		cube black0 = (item2 + item4) / (-N);   //��ɫ�ݶ���
		cube red0 = (tmp % (2 * (sum(sum((*cache[0])[k], 0), 1) / (H*W) + (*mean_))));   //��ɫ�ݶ���
		cube black_(H, W, C, fill::zeros);  //�㲥��ĺ�ɫ�ݶ���
		cube red_(H, W, C, fill::zeros);	 //�㲥��ĺ�ɫ�ݶ���
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

	// ��ʼ��w��b��w��b����ÿ��ͨ��һ����
	if (!in[1])
	{
		in[1].reset(new Blob(1, C, 1, 1, TONES));
		cout << "initLayer: " << lname << "  Init  ��  with Ones ;" << endl;
	}
	if (!in[2])
	{
		in[2].reset(new Blob(1, C, 1, 1, TZEROS));
		cout << "initLayer: " << lname << "  Init  ��  with Zeros ;" << endl;
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
    // ������
	out.reset(new Blob(in[0]->size(), TZEROS));

	int N = in[0]->get_N();
	int C = in[0]->get_C();
	int H = in[0]->get_H();
	int W = in[0]->get_W();

	// ͨ���㲥��ʵ�ֲ�����Ӧ��ƥ��
	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> beta(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c)
	{
		(*gamma)[0].slice(c).fill(as_scalar((*in[1])[0].slice(c)));
		(*beta)[0].slice(c).fill(as_scalar((*in[2])[0].slice(c)));
	}
	// ����ƽ�ƺ�����
	for (int n = 0; n < N; ++n)
		(*out)[n] = (*gamma)[0] % (*in[0])[n] + (*beta)[0];  //out  = �� * in    +  ��

	return;
}

void ScaleLayer::backward(const shared_ptr<Blob>& din,
												const vector<shared_ptr<Blob>>& cache,
												vector<shared_ptr<Blob>>& grads,
												const Param& param)
{
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));//dx  
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));//d��
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));//d��
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	//1. �� ͨ���㲥��ɳߴ�ƥ��
	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c)
		(*gamma)[0].slice(c).fill(as_scalar((*cache[1])[0].slice(c)));  //��Ϊdx  = din % ��  �����Ԧ� ��Ҫ�㲥��ɳߴ�ƥ��


	//2. ��������ݶ�
	shared_ptr<Blob> dgamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> dbeta(new Blob(1, C, H, W, TZEROS));
	for (int n = 0; n < N; ++n)
	{
		(*grads[0])[n] = (*din)[n] % (*gamma)[0];			 // dx  = din % ��        ��N,C,H,W��
		(*dgamma)[0] += (*din)[n] % (*cache[0])[n];    // d��  = din % x        ��1, C,H,W��   ע����ߵĲ�ͬ�������ݶ��ۼ�
		(*dbeta)[0] += (*din)[n];									 // d��  = din               ��1, C,H,W��   ע����ߵĲ�ͬ�������ݶ��ۼ�
	}
	(*grads[1])[0] = sum(sum((*dgamma)[0], 0), 1) / N;    //�ݶȺ�����ƽ��d��
	(*grads[2])[0] = sum(sum((*dbeta)[0], 0), 1) / N;         //�ݶȺ�����ƽ��d��

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
	dout.reset(new Blob(N, C, Hx, Wx));   //��N,C,1,1��
	double loss_ = 0;
	for (int i = 0; i < N; ++i)
	{
	    // softmax
		cube prob = arma::exp((*in[0])[i]) / arma::accu(arma::exp((*in[0])[i]));
		// �������ۼ�
		loss_ += (-arma::accu((*in[1])[i] % arma::log(prob )));
		// �ݶȼ���
		(*dout)[i] = prob - (*in[1])[i];
	}
	loss = loss_ / N;   //��ƽ����ʧ
	return;
}

void SVMLossLayer::hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
	if (dout)
		dout.reset();
	//-------step1.��ȡ��سߴ�
	int N = in[0]->get_N();        //����Blob��cube��������batch����������
	int C = in[0]->get_C();         //����Blobͨ����
	int Hx = in[0]->get_H();      //����Blob��
	int Wx = in[0]->get_W();    //����Blob��
	assert(Hx == 1 && Wx == 1);
	dout.reset(new Blob(N, C, Hx, Wx));   //��N,C,1,1��
	double loss_ = 0;
	double delta = 0.2;
	for (int i = 0; i < N; ++i)
	{
		//(1).������ʧ
		int idx_max = (*in[1])[i].index_max();
		double positive_x = (*in[0])[i](0, 0, idx_max);
		cube tmp = ((*in[0])[i] - positive_x + delta);           //����hinge loss��ʽ
		tmp(0, 0, idx_max) = 0;                                              //�޳���ȷ�������ֵ
		tmp.transform([](double e) {return e > 0 ? e : 0; });  //��max()�������õ������������ʧ
		loss_ +=arma::accu(tmp);  //�õ�����������ʧ��

		//(2).�����ݶ�
		tmp.transform([](double e) {return e ? 1 : 0; });
		tmp(0,0,idx_max)= -arma::accu(tmp);
		(*dout)[i]=tmp;

	}
	loss = loss_ / N;   //��ƽ����ʧ
	return;
}