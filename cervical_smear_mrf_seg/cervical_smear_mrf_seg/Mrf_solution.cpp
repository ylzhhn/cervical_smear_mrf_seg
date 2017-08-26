#include "Mrf_solution.h"
MrfSolve::MrfSolve(){}
MrfSolve::~MrfSolve(){}

//input function
void MrfSolve::SetInData(string SuDataFile,
	string SuFlagFile, string InitLabelFile,
	string GraphLinkFile,
	double T, double Beta,
	int MaxIter, int ClassNum)
{

	//load data
	suData_mat.load(SuDataFile);
	suFlag_mat.load(SuFlagFile);
	initLabel_list.load(InitLabelFile);
	Load2Vec(GraphLinkFile);// load graphLink

	//data format preprocess
	graphLink = deleteNoneNeighbor();

	//mrf parameters
	t = T;
	beta = Beta;
	maxIter = MaxIter;
	classNum = ClassNum;

	suPixelNum = suData_mat.n_rows;
	Featdim = suData_mat.n_cols - 3;

	mrfLabel_list = initLabel_list.col(1);
	suGtList = suData_mat.col(suData_mat.n_cols - 1);
	slicSeqNo_initNo();

	CalculateMeanSigma(suGtList);
}

//solution main function
void MrfSolve::MakeSolve()
{
	double E(0);//global energy
	double LocalE(0);//local energy
	int IterCount(0);//iterated  times
	double detaE;//difference of between two iterations
	//init energy
	double E_old = CalculateEnergy();

	do
	{
		detaE = 0;
	
		for (int i = 0; i < suPixelNum; i++)
		{
			if (suFlag_mat(i, 1) == 1)
			{
				for (int k = 0; k < classNum; k++)
				{
					double tmp1 = CalculateLocalEnergy(i, mrfLabel_list(i, 0));
					double tmp2 = CalculateLocalEnergy(i, k + 1);
					if (tmp1 > tmp2)
						mrfLabel_list(i, 0) = k + 1;
				}
			}

		}

		E = CalculateEnergy();
		detaE += fabs(E_old - E);
		E_old = E;
		
		IterCount++;
	} while (detaE > t && IterCount< maxIter);

}

//output function
arma::mat MrfSolve::GetRst()
{
	return mrfLabel_list;
}

//assistent function
void MrfSolve::Load2Vec(string graphLinkFile)
{
	ifstream In(graphLinkFile);
	string line;
	while (getline(In, line))
	{
		vector<int> NumList;
		NumList = CutLine2Num(line);
		graphLink.push_back(NumList);
		NumList.erase(NumList.begin(), NumList.begin() + NumList.size());
	}

	In.close();
}

vector<int> MrfSolve::CutLine2Num(string line)
{
	vector<int> NumList;
	size_t found = line.find_first_of(" ");
	size_t Numbegin = 0;
	size_t Numend = found;
	
	while (found != std::string::npos)
	{
		string curIndex = line.substr(Numbegin, Numend - Numbegin);
		int n = atof(curIndex.c_str());
		NumList.push_back(n);
		
		Numbegin = Numend + 1;
		found = line.find_first_of(" ", found + 1);
		Numend = found;
	}
	return NumList;
}

vector<vector<int>> MrfSolve::deleteNoneNeighbor()
{
	vector<vector<int>> rst;
	for (int i = 0; i < graphLink.size(); i++)
	{
		vector<int> tmpRow;
		tmpRow = graphLink[i];
		if (tmpRow.size() != 1)
			rst.push_back(tmpRow);
	}
	return rst;
}

void MrfSolve::CalculateMeanSigma(mat cur)
{
	arma::cube tmpS(Featdim, Featdim, classNum);
	for (int i = 0; i < classNum; i++)
	{
		uvec q1 = find(cur == i + 1);
		mat classI;
		for (int j = 0; j < q1.size(); j++)
		{
			mat tmpRow = (suData_mat.row(q1[j])).cols(2, Featdim + 1);
			classI.insert_rows(j, tmpRow);
		}

		Mu.insert_rows(i, mean(classI));
		tmpS.slice(i) = cov(classI);

	}
	Sigma = tmpS;
}

double MrfSolve::CalculateEnergy()
{
	double E(0);
	double singletons(0);
	double doubletons(0);
	for (int i = 0; i < suPixelNum; i++)
	{
		int label = mrfLabel_list(i, 0);
		
		singletons += Singleton(i, label);
		
		doubletons += Doubleton(i, label);
	}

	E += singletons + doubletons / 2;
	return E;
}

double MrfSolve::Singleton(int index, int label)
{
	
	mat tmp4 = randu<mat>(1, 13);
	mat tmp5;
	mat tmp1;
	double tmp3;
	double tmp2;

	mat E;
	mat y_s = suData_mat.row(index);
	y_s = y_s.cols(2, suData_mat.n_cols - 2);
	mat curMean = Mu.row(label - 1);
	
	mat curSigma = Sigma.slice(label - 1);
	mat diff = y_s - curMean;

	double det_value = det(curSigma);
	if (det_value == 0)
	{
		//mat trueLabel = suData_mat.col(suData_mat.n_cols - 1);
		CalculateMeanSigma(suGtList);
		curSigma = Sigma.slice(label - 1);

		
	}

	tmp4 = inv(curSigma);
	tmp3 = det(curSigma);
	tmp2 = log(tmp3);

	E = diff* tmp4 * diff.t() + tmp2;

	return E(0, 0);
}

double MrfSolve::Doubleton(int index, int label)
{
	double E(0);
	
	int slicIndex = suData_mat(index, 1);
	vector<int> tmpRow = graphLink[index];
	
	int orderNo = usingNoFindlabel(tmpRow[0]);
	int slicLabel = mrfLabel_list(orderNo, 0);
	for (int i = 1; i < tmpRow.size(); i++)
	{
		
		int orderNeighborNo = usingNoFindlabel(tmpRow[i]);
		int neighborL = mrfLabel_list(orderNeighborNo, 0);

		if (neighborL != label)
			E += beta;
	}
	return E;
}


int  MrfSolve::usingNoFindlabel(int slicNo)
{
	
	return (InitLDict.find(slicNo)->second);
}
void MrfSolve::slicSeqNo_initNo()
{
	//map<int, int> InitLDict;
	arma::mat DictM = suData_mat.cols(0, 1);
	for (int i = 0; i < DictM.n_rows; i++)
	{
		InitLDict.insert(std::pair<int, int>(DictM(i, 1), DictM(i, 0)));
	}
}
double MrfSolve::CalculateLocalEnergy(int indice, int label)
{
	double E_local(0);

	E_local = Singleton(indice, label) + Doubleton(indice, label);

	return E_local;
}
