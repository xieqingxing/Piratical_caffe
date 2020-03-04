#include "myBlob.hpp"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int h, const int w, int type) : N_(n), C_(c), H_(h), W_(w)
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	_init(N_, C_, H_, W_, type);
	
}

Blob::Blob(const vector<int> shape_, int type) : N_(shape_[0]), C_(shape_[1]), H_(shape_[2]), W_(shape_[3])
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	_init(N_, C_, H_, W_, type);
}

void Blob::_init(const int n, const int c, const int h, const int w, int type)
{

	if (type == TONES)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::ones));
		return;
	}
	if (type == TZEROS)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::zeros));
		return;
	}
	if (type == TDEFAULT)
	{
		blob_data = vector<cube>(n, cube(h, w, c));
		return;
	}
	if (type == TRANDU)
	{
		for (int i = 0; i < n; ++i)   //����n����������ֵ�����ȷֲ�����cube���ѵ���vector����
			blob_data.push_back(arma::randu<cube>(h, w, c)); //�ѵ�
		return;
	}
	if (type == TRANDN)
	{
		for (int i = 0; i < n; ++i)   //����n����������ֵ(��׼��˹�ֲ�����cube���ѵ���vector����
			blob_data.push_back(arma::randn<cube>(h, w, c)); //�ѵ�
		return;
	}

}

vector<int> Blob::size() const
{
	vector<int> shape_{   N_,
										 C_,
										 H_,
										 W_ };
	return shape_;
}

void Blob::print(string str)
{
	assert(!blob_data.empty());  //���ԣ�   blob_data��Ϊ�գ�������ֹ����
	cout << str << endl;
	for (int i = 0; i < N_; ++i)  //N_Ϊblob_data��cube����
	{
		printf("N = %d\n", i);
		this->blob_data[i].print();//��һ��ӡcube������cube�����غõ�print()
	}
}

cube& Blob::operator[] (int i)
{
	return blob_data[i];
}

Blob& Blob::operator*= (const double k)
{
	for (int i = 0; i < N_; ++i)
	{
		blob_data[i] = blob_data[i] * k;   //����cube��ʵ�ֵ�*������
	}
	return *this;
}

Blob& Blob::operator= (double val)
{
	for (int i = 0; i < N_; ++i)
	{
		blob_data[i].fill(val);   //����cube��ʵ�ֵ�*������
	}
	return *this;
}

Blob operator*(Blob& A, Blob& B)  //��Ԫ�����ľ���ʵ�֣�����û�����޶����� (Blob& Blob::)������ʽ
{
	//(1). ȷ����������Blob�ߴ�һ��
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //���ԣ���������Blob�ĳߴ磨N,C,H,W��һ����
	}
	//(2). �������е�cube��ÿһ��cube����Ӧλ����ˣ�cube % cube��
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] =A[i] % B[i];
	}
	return C;
}

Blob operator*(double num, Blob& B)
{
	//�������е�cube��ÿһ��cube������һ����ֵnum
	int N =B.get_N();
	Blob out(B.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = num * B[i];
	}
	return out;
}

Blob operator/(Blob& A, Blob& B)
{
	//(1). ȷ����������Blob�ߴ�һ��
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //���ԣ���������Blob�ĳߴ磨N,C,H,W��һ����
	}
	//(2). �������е�cube��ÿһ��cube����Ӧλ�������cube / cube��
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] / B[i];
	}
	return C;
}

Blob operator/(Blob& A, double val)
{
	//(1). �������е�cube��ÿһ��cube������һ����
	int N = A.get_N();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = A[i] / val;
	}
	return out;
}

Blob operator+(Blob& A, Blob& B)
{
	//(1). ȷ����������Blob�ߴ�һ��
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //���ԣ���������Blob�ĳߴ磨N,C,H,W��һ����
	}
	//(2). �������е�cube��ÿһ��cube����Ӧλ����ӣ�cube + cube��
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] + B[i];
	}
	return C;
}

Blob operator+(Blob& A, double val)
{
	//�������е�cube��ÿһ��cube������һ����ֵ
	int N = A.get_N();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = A[i] + val;   
	}
	return out;
}

Blob sqrt(Blob& A)
{
	int N = A.get_N();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = arma::sqrt(A[i]);
	}
	return out;
}

Blob square(Blob& A)
{
	int N = A.get_N();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = arma::square(A[i]);
	}
	return out;
}

double accu(Blob& A)
{
	int N = A.get_N();
	double result=0;
	for (int i = 0; i < N; ++i)
	{
		result += arma::accu(A[i]);
	}
	return result;
}

vector<cube>& Blob::get_data()
{
	return blob_data;
}

Blob Blob::subBlob(int low_idx, int high_idx)
{
	//������ [0,1,2,3,4,5]  -> [1,3)  -> [1,2]
	if (high_idx > low_idx)
	{
		Blob tmp(high_idx - low_idx, C_, H_, W_);  // high_idx > low_idx
		for (int i = low_idx; i < high_idx; ++i)
		{
			tmp[i - low_idx] = (*this)[i];
		}
		return tmp;
	}
	else
	{
		// low_idx >high_idx
		//������ [0,1,2,3,4,5]  -> [3,2)-> (6 - 3) + (2 -0) -> [3,4,5,0]
		Blob tmp(N_ - low_idx + high_idx, C_, H_, W_);
		for (int i = low_idx; i < N_; ++i)   //�ֿ����ν�ȡ���Ƚ�ȡ��һ��
		{
			tmp[i - low_idx] = (*this)[i];
		}
		for (int i = 0; i < high_idx; ++i)   //�ֿ����ν�ȡ���ٽ�ȡѭ������0��ʼ�����
		{
			tmp[i + N_ - low_idx] = (*this)[i];
		}
		return tmp;
	}
}

Blob Blob::pad(int pad, double val)
{
	assert(!blob_data.empty());
	// ����һ����Blob�����
	Blob padX(N_, C_, H_ + 2 * pad, W_ + 2 * pad);
	padX = val;
	for (int n = 0; n < N_; ++n)
	{
		for (int c = 0; c < C_; ++c)
		{
			for (int h = 0; h < H_; ++h)
			{
				for (int w = 0; w < W_; ++w)
				{
					padX[n](h + pad, w + pad, c) = blob_data[n](h, w, c);
				}
			}
		}
	}
	return padX;

}

Blob Blob::deletePad(int pad)
{
	assert(!blob_data.empty());   //���ԣ�Blob����Ϊ��
	Blob out(N_, C_, H_ - 2 * pad, W_ - 2 * pad);
	for (int n = 0; n < N_; ++n)
	{
		for (int c = 0; c < C_; ++c)
		{
			for (int h = pad; h < H_-pad; ++h)
			{
				for (int w = pad; w < W_-pad; ++w)
				{
					//ע�⣬out�������Ǵ�0��ʼ�ģ�����Ҫ��ȥpad
					out[n](h - pad, w - pad, c) = blob_data[n](h, w, c);
				}
			}
		}
	}
	return out;
}

void Blob::maxIn(double val)
{
	assert(!blob_data.empty());
	for (int i = 0; i < N_; ++i)
	{
        // .transform(lambda_function)
		blob_data[i].transform([val](double e){return e>val ? e : val; });
		// Ϊ�˷�ֹ������̫�󣬽�������
		double clipped_num = 6.0;
		blob_data[i].transform([clipped_num](double e){return e>clipped_num ? clipped_num : e; });
	}
	return;
}

void Blob::convertIn(double val)
{
    // ��������ɵ�0,1�����drop_rate�ȴ�С����ɸѡ
	assert(!blob_data.empty());
	for (int i = 0; i < N_; ++i)
	{
		blob_data[i].transform([val](double e){return e<val ? 0 : 1; });
	}
	return;
}