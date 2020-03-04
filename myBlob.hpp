#ifndef __MYBLOB_HPP__
#define __MYBLOB_HPP__
#include <vector>
#include <armadillo>

using std::vector;
using arma::cube;
using std::string;

enum FillType
{

	TONES = 1,  //cube����Ԫ�ض����Ϊ1
	TZEROS = 2, //cube����Ԫ�ض����Ϊ0
	TRANDU = 3,  //��Ԫ������Ϊ[0,1]�����ھ��ȷֲ������ֵ
	TRANDN = 4,  //ʹ�æ�= 0�ͦ�= 1�ı�׼��˹�ֲ�����Ԫ��
	TDEFAULT = 5
	
};


//Blob a;
//Blob a(10,3,3,3,TONES);
class Blob
{
public:
	Blob() : N_(0), C_(0), H_(0), W_(0)
	{}
	Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);
	Blob(const vector<int> shape_, int type = TDEFAULT);
	
	void print(string str = "");  
	vector<cube>& get_data();
	cube& operator[] (int i);
	Blob& operator= (double val);
	Blob& operator*= (const double i);
	friend Blob operator*(Blob& A, Blob& B);
	friend Blob operator/(Blob& A, Blob& B);
	friend Blob operator/(Blob& A, double val);
	friend Blob operator*(double num, Blob& B);
	friend Blob operator+(Blob& A, Blob& B);
	friend Blob operator+(Blob& A, double val);
	friend Blob sqrt(Blob& A);
	friend Blob square(Blob& A);
	friend double accu(Blob& A);
	Blob subBlob(int low_idx, int high_idx);
	Blob pad(int pad,double val=0);
	Blob deletePad(int pad);
	void maxIn(double val=0.0);
	void convertIn(double val = 0.0);
	vector<int> size() const;
	inline int get_N() const
	{
		return N_;
	}
	inline int get_C() const
	{
		return C_;
	}
	inline int get_H() const
	{
		return H_;
	}
	inline int get_W() const
	{
		return W_;
	}
private:
	void _init(const int n, const int c, const int h, const int w, int type);

private:
	int N_;
	int C_;
	int H_;
	int W_;
	vector<cube> blob_data;
};


#endif // __MYBLOB_HPP__
