n#include<stdio.h>
#include<cstdlib>
#include<iostream>
#include<string>
#include<fstream>
#include<dirent.h>
#include<stdlib.h>
#include <tr1/unordered_map>
#include<vector>
#include <sstream>
#include <google/sparse_hash_map>
#include <math.h>
#include <iomanip>
#include <armadillo>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace arma;

using google::sparse_hash_map;
using std::cout;
using std::endl;
using tr1::hash;  // or ext::hash, or maybe tr1::hash, depending on your OS


struct eqhash
{
   bool operator()(const long int s1, const long int s2) const
   {
      return (s1 == s2);
   }
};

struct idealArt
{
    long int artId;
    double ctr;

    idealArt()
    {
        artId=0;
        ctr=0.0;
    }
};

struct ucbArt
{
    long int artId;
    double ucbScore;
    ucbArt()
    {
        artId=0;
        ucbScore=0.0;
    }
};

struct gpArt
{
    long int artId;
    double gpucb;
    gpArt()
    {
        artId=0;
        gpucb=0.0;
    }
};

bool compareCtr(const idealArt* a, const idealArt* b)
{
    return a->ctr > b->ctr;
}

bool compareUcb(const ucbArt* a, const ucbArt* b)
{
    return a->ucbScore > b->ucbScore;
}

bool compareGp(const gpArt* a, const gpArt* b)
{
    return a->gpucb > b->gpucb;
}

struct outStatsLine
{
    long int shows;
    long int clicks;
    double ctr;
};

vector<struct outStatsLine*> idealStats;
vector<struct outStatsLine*> ucbStats;
vector<struct outStatsLine*> gpStats;
vector<struct outStatsLine*> bucbStats;
vector<struct outStatsLine*> randomStats;
vector<struct outStatsLine*> linUcbStats;
vector<struct outStatsLine*> linRelUcbStats;


void loadFiles();

struct artFeat
{
    long artId;
    double feats[6];
    double artShows;
    double artClicks;
    double ctr;
    double showsIdeal;
    double clicksIdeal;
    double showsUCB[10];
    double clicksUCB[10];
    double showsGP;
    double clicksGP;
    double showsBucb;
    double clicksBucb;
    double showsLinUcb;
    double clicksLinUcb;
    double showsLinRelUcb;
    double clicksLinRelUcb;
    double idealCtr;
    double ucbScore;
    double gpUcb;
    double bucb;

    artFeat()
    {
        artId = 0;
        artShows = 0.0 ;
        artClicks = 0.0;
        ctr = 0.0;
        showsIdeal = 0.0;
        clicksIdeal = 0.0;
        showsGP = 0.0;
        clicksGP = 0.0;
        showsBucb = 0.0;
        clicksBucb = 0.0;
        showsLinUcb = 0;
        clicksLinUcb= 0;
        showsLinRelUcb = 0;
        clicksLinRelUcb = 0;
        idealCtr = 0.0;
        ucbScore = 0.0;
        gpUcb = 0.0;
        bucb = 0.0;
    };

};

//global variables
long int linesDone = 0;
long int totalShowsUCB = 0;
long int numShowsUCB[10];
long int numShowsGP = 0;
long int numShowsIdeal = 0;
long int numShowsBucb = 0;
long int numShowsRandom = 0;
long int numShowsLinUcb= 0;
long int numShowsLinRelUcb = 0;

double totalClicksUCB = 0;
double numClicksUCB[10];
double numClicksGP = 0;
double numClicksIdeal = 0;
double numClicksBucb = 0;
double numClicksRandom = 0;
double numClicksLinUcb = 0;
double numClicksLinRelUcb = 0;

double numShowsTotal = 0;
double numClicksTotal = 0;
int batchSize;
int scoringRule;
int betaType;

double centroid[10][6] = {0.0};

// 10 predefined centroids for the articles (the articles are clustered in advance???)
mat centroidsArma = zeros<mat>(10,6);

vector<double> showsIdeal;
vector<double> clicksIdeal;

vector<double> showsUCB;
vector<double> clicksUCB;

vector<double> showsGP;
vector<double> clicksGP;

vector<double> showsActual;
vector<double> clicksActual;

vector<double> showsBucb;
vector<double> clicksBucb;

vector<double> showsLinUcb;
vector<double> clicksLinUcb;

vector<double> showsLinRelUcb;
vector<double> clicksLinRelUcb;

// Position weights - will check what the positioning is for.
vector<double> posWeight;

sparse_hash_map<long int, struct artFeat*, hash<long int>, eqhash> artHash1;
sparse_hash_map<long int, struct artFeat*, hash<long int>, eqhash>::iterator it;

sparse_hash_map<long int, long int, hash<long int>, eqhash> articleBirth;

// this is <article , list<ctr> > where list is order of clusters and cluster/article ctr
sparse_hash_map<long int, vector<double>, hash<long int>, eqhash> idealCtr;
// same as above but ucb instead of ideal ctr
sparse_hash_map<long int, vector<double>, hash<long int>, eqhash> currentUcb;

vector<struct idealArt*> sortedCtr[10];
vector<struct ucbArt*> sortedUcb[10];

//GP specific constructs
sparse_hash_map<long int, vector<double>, hash<long int>, eqhash> currentGp;
vector<struct gpArt*> sortedGp[10];
mat kernel;
vector<long int> gpOrder;
mat means;
mat numPulls;
mat fb;
const double noiseVar = 0.1;
vec beta;
umat indices;
mat ucbScore;
int numElem;
vec artAge;
//GP specific constructs

//BUCB specific constructs
sparse_hash_map<long int, vector<double>, hash<long int>, eqhash> currentBucb;
vector<struct gpArt*> sortedBucb[10];
mat meansBucb;
mat numPullsBucb;
mat fbBucb;
vec betaBucb;
umat indicesBucb;
mat bucbScore;
vec artAgeBucb;
//artHash1.set_deleted_key(NULL);

//LInUCB constructs
double alphaLin = 0.5;
mat A0;
mat A0Inv;
vec b0;
vec betaHat;

sparse_hash_map<long int, mat, hash<long int>, eqhash> article_Aa;
sparse_hash_map<long int, mat, hash<long int>, eqhash> article_Ba;
sparse_hash_map<long int, vec, hash<long int>, eqhash> article_ba;
sparse_hash_map<long int, mat, hash<long int>, eqhash> article_AaInv;
sparse_hash_map<long int, vec, hash<long int>, eqhash> article_thetaA;
sparse_hash_map<long int, double, hash<long int>, eqhash> article_uncert;
sparse_hash_map<long int, double, hash<long int>, eqhash> linUcbScore;

//LinRelUCB constructs
double alphaRelLin = 0.5;
mat ARel;
mat AInvRel;
vec bRel;
vec thetaRel;
sparse_hash_map<long int, double, hash<long int>, eqhash> article_uncert_Rel;
sparse_hash_map<long int, double, hash<long int>, eqhash> linRelUcbScore;

struct line
{
    long int timestamp;
    long int dispArt;
    bool click;
    vector<double> userFeat;
    unsigned int clusterId;
    int numArts;
    vector<long int> artId;
    vector<struct artFeat*> artFeatList;
    ~line()
    {
        for(int i = 0; i< this->numArts; i++)
        {
            delete this->artFeatList[i];
        }
        this->artFeatList.clear();
        this->artId.clear();
    }
};

/* Find the nearest cluster for per user */
unsigned int findNearestCentroid(vector<double> user)
{
    vec distances = zeros<vec>(10);
    vec userFeat = zeros<vec>(6);
    userFeat = conv_to<vec>::from(user);

    for(int i = 0; i<10;i++)
    {
        distances(i) = norm((userFeat.t()-centroidsArma.row(i)),2);
    }
    unsigned int index;
    double min_val = distances.min(index);
    return index;
}


struct line* extractFromLogLine(struct line* currLine, const string logLineStr)
{
    char tempChar, tempChar2;
    int numArts = 0;
    long int artId;
    std::stringstream lineStream(logLineStr);
    lineStream >> currLine->timestamp;
    lineStream >> currLine->dispArt;
    lineStream >> currLine->click;
    string userString;
    lineStream >> userString;
    string featStr;
    std::string featValStr;

    for(int i = 0; i < 6; i++)
    {
        currLine->userFeat.push_back(0);
    }

    // Read the user features
    for(int iter = 0 ; iter < 6 ; iter++)
    {
        lineStream >> featStr;
        featValStr = featStr.substr(2);
        // Adjust the index of the features
        currLine->userFeat[((iter+1)%6)] = atof(featValStr.c_str());
    }

    currLine->clusterId = findNearestCentroid(currLine->userFeat);

    string tempLineStr;
    while(lineStream >> tempLineStr)
    {
        featValStr = tempLineStr.substr(1);
        long int availArtId = atoi(featValStr.c_str());
        currLine->artId.push_back(availArtId);
        numArts++;

        struct artFeat* artFeat1 = new artFeat();
        for(int iter = 0 ; iter < 6 ; iter++)
        {
            lineStream >> featStr;
            if(atoi(featStr.substr(0,1).c_str()) > 6)
            {
                artFeat1->feats[1] = 1.0;
                break;
            }
            if(artHash1.find(availArtId) == artHash1.end())
            {
                featValStr = featStr.substr(2);
                artFeat1->feats[((iter+1)%6)] = atof(featValStr.c_str());
            }
        }

        currLine->artFeatList.push_back(artFeat1);

    }
    currLine->numArts = numArts;
    lineStream << "";
    lineStream.clear();
    return currLine;
}

void dumpVector(vector<struct outStatsLine*>& inVec, string filename)
{
    ofstream outfile;
    outfile.open(filename.c_str(), fstream::out);
    vector<struct outStatsLine*>::iterator it;

    for(it = inVec.begin(); it != inVec.end(); it++)
    {
        outfile << " " << ((struct outStatsLine*)(*it))->shows;
        outfile << " " << ((struct outStatsLine*)(*it))->clicks;
        outfile << " " << fixed << ((struct outStatsLine*)(*it))->ctr << endl;
    }
    outfile.close();
}

void initialize()
{
    /* Read and initialize preprocessed user centers from file */
    ifstream centroidFile;
    centroidFile.open("userFeatCenters.txt",fstream::in);
    int clusNum = 0;
    string line;
    while(getline(centroidFile, line))
    {
        std::stringstream lineStream(line);
        for(int i = 0; i < 6; i++)
        {
            lineStream >> centroid[clusNum][i];
            centroidsArma(clusNum,i) = centroid[clusNum][i];
        }
        centroidsArma.row(clusNum).print(cout,"centre");
        clusNum++;
    }
    centroidFile.close();


    ifstream ctrFile;
    /* Read from a file the sorted pair of cluster id, article id, ctr for that cluster article id. 
    Potentially the articles are also clustered in 10 clusters in advance. But unclear if that's the case.
    */
    ctrFile.open("sortedCtrCluster", fstream::in);
    while (getline(ctrFile, line))
    {
        int clusterId;
        long int artId;
        double ctr;
        struct idealArt* iterArt = new idealArt();
        struct ucbArt* ucbArt1 = new ucbArt();
        std::stringstream lineStream(line);
        lineStream >> clusterId;
        lineStream >> artId;
        lineStream >> ctr;

        iterArt->artId = artId;
        iterArt->ctr = ctr;

        ucbArt1->artId = artId;
        ucbArt1->ucbScore = (rand()/RAND_MAX);
        
        idealCtr[artId].push_back(ctr);
        currentUcb[artId].push_back((rand()/RAND_MAX));

        sortedCtr[clusterId].push_back(iterArt);
        sortedUcb[clusterId].push_back(ucbArt1);
    }
    ctrFile.close();

    /* Reading position weights. This is based on the position where the article is shown. */
    ifstream posFile;
    posFile.open("posWeights", fstream::in);
    int pos = 0;
    while(getline(posFile,line))
    {
        std::stringstream linestream(line);
        string doubleString;
        linestream >> doubleString;
        posWeight.push_back(atof(doubleString.c_str()));
        pos++;
    }
    posFile.close();

    /* Similar what order for GP is necessary */
    ifstream gpOrderFile;
    string line2;
    gpOrderFile.open("gpOrder.dat",fstream::in);
    pos = 0;
    while(getline(gpOrderFile,line2))
    {
        long int iterId = atoi(line2.c_str());
        gpOrder.push_back(iterId);
        pos++;
    }
    gpOrderFile.close();

    //GP initialize
    numElem = gpOrder.size();

    means = 0.05 * ones<mat>(numElem, 10);
    numPulls = 2 * ones<mat>(numElem, 10);
    fb =    0.05 * ones<mat>(numElem, 10);

    // Necessary for beta calculation
    artAge =    zeros<vec>(numElem);
    beta =   2 * ones<vec>(numElem);

    ucbScore = means;
    for(int i = 0; i < 10; i++)
    {
        vector<long int>::iterator idIter;
        int ucbScoreIter;
        for(idIter = gpOrder.begin(), ucbScoreIter = 0; idIter != gpOrder.end(); idIter++, ucbScoreIter++)
        {
            struct gpArt* gpArt1 = new gpArt();
            gpArt1->artId = *idIter;
            gpArt1->gpucb = ucbScore(ucbScoreIter, i);
            sortedGp[i].push_back(gpArt1);
            currentGp[*idIter].push_back(ucbScore(ucbScoreIter, i));
        }
    }

    //BUCB initialize
    meansBucb = 0.05 * ones<mat>(numElem,10);
    numPullsBucb = 2 * ones<mat>(numElem,10);
    fbBucb =    0.05 * ones<mat>(numElem,10);

    artAgeBucb =    zeros<vec>(numElem);
    betaBucb =   2 * ones<vec>(numElem);

    // This seems like a bug or the previous init is unnecessary
    meansBucb = means;
    bucbScore = means;

    for(int i = 0; i < 10; i++)
    {
        vector<long int>::iterator idIter;
        int bucbScoreIter;
        for(idIter = gpOrder.begin(), bucbScoreIter = 0; idIter != gpOrder.end(); idIter++, bucbScoreIter++)
        {
            struct gpArt* gpArt1 = new gpArt();
            gpArt1->artId = *idIter;
            gpArt1->gpucb = bucbScore(bucbScoreIter, i);
            sortedBucb[i].push_back(gpArt1);
            currentBucb[*idIter].push_back(bucbScore(bucbScoreIter, i));
        }
    }

    //LinUCB initialize
     A0.eye(36, 36);
     A0Inv.eye(36, 36);
     b0 = zeros<vec>(36);

     vector<long int>::iterator idIter;
     for(idIter = gpOrder.begin(); idIter != gpOrder.end(); idIter++)
     {
         mat Aa;
         Aa.eye(6,6);
         mat Ba;
         Ba.zeros(6,36);
         vec ba;
         ba.zeros(6);
         mat AaInv;
         AaInv.eye(6,6);
         vec thetaA;
         thetaA.zeros(6);
         article_Aa[*idIter] = Aa;
         article_Ba[*idIter] = Ba;
         article_ba[*idIter] = ba;
         article_AaInv[*idIter] = AaInv;
         article_thetaA[*idIter] = thetaA;
         article_uncert[*idIter] = 10;
         linUcbScore[*idIter] = 0;
     }
     betaHat = A0Inv * b0;

     //LinRel initialize
     ARel.eye(36, 36);
     AInvRel.eye(36, 36);
     bRel = zeros<vec>(36);
     thetaRel = zeros<vec>(36);

     for(idIter = gpOrder.begin(); idIter != gpOrder.end(); idIter++)
     {
         article_uncert_Rel[*idIter] = 10;
         linRelUcbScore[*idIter] = 0;
     }

    cout << "init done with " << currentGp.size() << " entries" << endl;
}

/* Gets the ideal score for the currently available articles. Sorts them according to the ideal score and then gets the top batchSize ones. */
void idealSelect(struct line* line1, vector<long int>& selected, int flag)
{
    vector<struct idealArt*>::iterator ctrIter;
    vector<long int>::iterator idIter;

    unsigned int clusterId = line1->clusterId;
    for(idIter = line1->artId.begin(), ctrIter = sortedCtr[clusterId].begin(); idIter != line1->artId.end(); idIter++, ctrIter++)
    {
        ((struct idealArt*)(*ctrIter))->artId = *idIter;
        ((struct idealArt*)(*ctrIter))->ctr = idealCtr[*idIter][clusterId];
    }
    
    std::sort(sortedCtr[clusterId].begin(), sortedCtr[clusterId].begin()+line1->numArts, &compareCtr);
    
    int iter = 0;
    for(ctrIter = sortedCtr[clusterId].begin(); iter<batchSize; ctrIter++, iter++)
    {
        selected.push_back((((struct idealArt*)(*ctrIter))->artId));
    }
}

/* Gets the UCB score for the currently available articles. Sorts them according to the UCB score and then gets the top batchSize ones. */
void ucbSelect(struct line* line1, vector<long int>& selected)
{
    vector<struct ucbArt*>::iterator ucbIter;
    vector<long int>::iterator idIter;

    unsigned int clusterId = line1->clusterId;

    for(idIter = line1->artId.begin(),ucbIter = sortedUcb[clusterId].begin();idIter!=line1->artId.end();idIter++,ucbIter++)
    {
        ((struct ucbArt*)(*ucbIter))->artId = *idIter;
        ((struct ucbArt*)(*ucbIter))->ucbScore = currentUcb[*idIter][clusterId];
    }

    std::sort(sortedUcb[clusterId].begin(), sortedUcb[clusterId].begin() + line1->numArts, &compareUcb);

    int iter = 0;
    for(ucbIter = sortedUcb[clusterId].begin(); iter<batchSize; ucbIter++, iter++)
    {
        selected.push_back((((struct ucbArt*)(*ucbIter))->artId));
    }
}

vec setBeta(vec & artAgeLocal)
{
    if(betaType == 1)
    {
        beta = 0.5 * ones<vec>(numElem);
    }

    if(betaType == 2)
    {
        beta= 0.8 * ones<vec>(numElem);
    }

    if(betaType == 3)
    {
         vec tempbeta1 = max(join_rows(0.3*ones<vec>(numElem),0.4*log(artAgeLocal/(5000000/5))),1);
        beta = max(join_rows(exp(-1*(artAgeLocal)/(5000000/5)),tempbeta1),1);
    }

    if(betaType == 4)
    {
        vec tempbeta1 = max(join_rows(0.6*ones<vec>(numElem),0.6*log(artAgeLocal/(5000000/5))),1);
        beta = max(join_rows(exp(-1*(artAgeLocal)/(5000000/5)),tempbeta1),1);
    }

    if(betaType ==5)
    {
        beta = max(join_rows(0.6*ones<vec>(numElem),0.6*log(artAgeLocal/(5000000/5))),1);
    }
    return beta;

}

/* Same as all selects. */
void gpSelect(struct line* line1, vector<long int>& selected)
{
    vector<struct gpArt*>::iterator gpIter;
    vector<long int>::iterator idIter;

    unsigned int clusterId = line1->clusterId;
    for(idIter = line1->artId.begin(), gpIter = sortedGp[clusterId].begin(); idIter != line1->artId.end(); idIter++, gpIter++)
    {
        ((struct gpArt*)(*gpIter))->artId = *idIter;
        ((struct gpArt*)(*gpIter))->gpucb = currentGp[*idIter][clusterId];
    }
    std::sort(sortedGp[clusterId].begin(), sortedGp[clusterId].begin() + line1->numArts, &compareGp);

    int iter = 0;
    for(gpIter = sortedGp[clusterId].begin(); iter < batchSize; gpIter++, iter++)
    {
        selected.push_back((((struct gpArt*)(*gpIter))->artId));
    }
}

void bucbSelect(struct line* line1, vector<long int>& selected, int flag)
{
    vector<struct gpArt*>::iterator gpIter;
    vector<long int>::iterator idIter;

    unsigned int clusterId = line1->clusterId;
    artAgeBucb = artAgeBucb + ones<vec>(numElem); 

    if(flag == 1)
    {
        vector<unsigned int> posVec;
        for(vector<long int>::iterator it = line1->artId.begin(); it != line1->artId.end(); it++)
        {
            unsigned int currPos = std::find(gpOrder.begin(), gpOrder.end(), *it) - gpOrder.begin();
            posVec.push_back(currPos);
            artAgeBucb(currPos) = linesDone - articleBirth[*it];
        }
        indicesBucb = conv_to<uvec>::from(posVec);
    }

    betaBucb = setBeta(artAgeBucb);
    int numSelected = 0;

    uvec id = zeros<uvec>(1);
    id = clusterId;

    vec numPullsTemp = numPullsBucb(indicesBucb,id);

    vec fbTemp = fbBucb(indicesBucb,id);
    mat kernel1Temp = kernel(indicesBucb, indicesBucb);
    vec betaTemp = betaBucb.elem(indicesBucb);
    vec meansBucbTemp = meansBucb(indicesBucb, id);
    vec bucbScoreTemp = bucbScore(indicesBucb, id);
    mat precBucb = diagmat(1/numPullsBucb.col(clusterId));
    mat precTemp = precBucb(indicesBucb, indicesBucb);   

    vector<unsigned int> selSoFar;
    uvec selectedSoFar;

    while(numSelected < batchSize)
    {
        unsigned int singleSelected;
        double tempTopScore = bucbScoreTemp.max(singleSelected);
        selSoFar.push_back(singleSelected);
        selectedSoFar = conv_to<uvec>::from(selSoFar);
        numSelected++;
        
        numPullsTemp(singleSelected) = numPullsTemp(singleSelected) + 1;
        fbTemp(singleSelected) = (fbBucb(singleSelected, clusterId) * numPullsTemp(singleSelected) + meansBucb(singleSelected, clusterId)) / (numPullsTemp(singleSelected)+1);
        precTemp(singleSelected,singleSelected) = 1/numPullsTemp(singleSelected);

        mat cd = chol(kernel1Temp + precTemp * noiseVar);

        mat tempDiv1 = solve(cd, kernel1Temp.t());
        mat tempDiv2 = solve(cd.t(), tempDiv1);
        mat temp3 = kernel1Temp - kernel1Temp * tempDiv2;
        vec sigma2 = temp3.diag();
        sigma2 = abs(sigma2);

        mat tempDiv3 = solve(cd,fbTemp);
        mat tempDiv4 = solve(cd.t(), tempDiv3);
        meansBucbTemp = kernel1Temp * tempDiv4;

        bucbScoreTemp = meansBucbTemp + betaTemp % sqrt(sigma2);
        vec lowScore(numSelected);
        lowScore.fill(-1000.0);
        bucbScoreTemp.elem(selectedSoFar) = lowScore;
    }

    uvec indexes = indicesBucb.elem(selectedSoFar);
    vector<long int>::iterator orderIter;
    for(int i=0;i<batchSize;i++)
    {
        selected.push_back(gpOrder[indexes(i)]);
    }

}

void linUcbSelect(struct line* line1, vector<long int>& selected)
{
    vec user = conv_to<vec>::from(vector<double>(line1->userFeat));
    double maxScore = -10000.0;
    long int chosenOne;
    for(vector<long int>::iterator it = line1->artId.begin(); it != line1->artId.end(); it++)
    {
         std::vector<double> artFeats(((struct artFeat *)(artHash1[*it]))->feats, ((struct artFeat *)(artHash1[*it]))->feats + 6);
         vec articleFeat = conv_to<vec>::from(artFeats);
         vec zta = reshape(articleFeat*user.t(), 36, 1, 0);
         vec uncertainty = zta.t()*A0Inv*zta - 2 * zta.t() * A0Inv * article_Ba[*it].t() * article_AaInv[*it] * user + user.t() * article_AaInv[*it] * user +
         user.t() * article_AaInv[*it] * article_Ba[*it] * A0Inv * article_Ba[*it].t() * article_AaInv[*it] * user;
         article_uncert[*it] = uncertainty(0);
         vec scoreLinUcb = zta.t() * betaHat + user.t() * article_thetaA[*it] + alphaLin * sqrt(article_uncert[*it]);
         linUcbScore[*it] = scoreLinUcb(0);

        if(linUcbScore[*it] > maxScore)
        {
            chosenOne = *it;
            maxScore = linUcbScore[*it];
        }
    }

    selected.push_back(chosenOne);
}

void updateLinUcb(vector<long int>& selected, vector<double>& showsVec, vector<double>& clicksVec, struct line* line1)
{
    vec user = conv_to<vec>::from(vector<double>(line1->userFeat));
    int pos=0;
    for(vector<long int>::iterator it= selected.begin();it!=selected.end();it++,pos++)
    {
        std::vector<double> artFeats(((struct artFeat *)(artHash1[*it]))->feats, ((struct artFeat *)(artHash1[*it]))->feats + 6);

        vec articleFeat = conv_to<vec>::from(artFeats);

        vec zta = reshape(articleFeat*user.t(),36,1,0);
        A0 = A0 + article_Ba[*it].t()*article_AaInv[*it]*article_Ba[*it];
        b0 = b0 + article_Ba[*it].t()*article_AaInv[*it]*article_ba[*it];
        article_Aa[*it] = article_Aa[*it] + user*user.t();

        article_AaInv[*it] = pinv(article_Aa[*it]);

        article_Ba[*it] = article_Ba[*it] + user*zta.t();
        article_ba[*it] = article_ba[*it] + clicksVec[pos]*(1/posWeight[pos])*user;
        A0 = A0 + zta*zta.t() - article_Ba[*it].t()*article_AaInv[*it]*article_Ba[*it];
        b0 = b0 + clicksVec[pos]*(1/posWeight[pos])*zta - article_Ba[*it].t()*article_AaInv[*it]*article_ba[*it];
        article_thetaA[*it] = article_AaInv[*it]* (article_ba[*it] - article_Ba[*it]*betaHat);

    }

    A0Inv = pinv(A0);
    betaHat = A0Inv*b0;

}

void linRelUcbSelect(struct line* line1, vector<long int>& selected)
{
    vec user = conv_to<vec>::from(vector<double>(line1->userFeat));
    double maxScore = -10000.0;
    long int chosenOne;
    for(vector<long int>::iterator it= line1->artId.begin();it!=line1->artId.end();it++)
    {

        std::vector<double> artFeats(((struct artFeat *)(artHash1[*it]))->feats, ((struct artFeat *)(artHash1[*it]))->feats + 6);
        vec articleFeat = conv_to<vec>::from(artFeats);
        vec zta = reshape(articleFeat*user.t(),36,1,0);
        thetaRel = AInvRel*bRel;
        vec linRelScore = thetaRel*zta + alphaRelLin*sqrt(zta.t()*AInvRel*zta);
        linRelUcbScore[*it] = linRelScore(0);

        if(linRelUcbScore[*it]>maxScore)
        {
            maxScore = linRelUcbScore[*it];
            chosenOne = *it;
        }

    }
    selected.push_back(chosenOne);
}

void updateLinRelUcb(vector<long int>& selected, vector<double>& showsVec, vector<double>& clicksVec, struct line* line1)
{
    vec user = conv_to<vec>::from(vector<double>(line1->userFeat));
    int pos=0;
    for(vector<long int>::iterator it= selected.begin();it!=selected.end();it++,pos++)
    {
        std::vector<double> artFeats(((struct artFeat *)(artHash1[*it]))->feats, ((struct artFeat *)(artHash1[*it]))->feats + 6);

         vec articleFeat = conv_to<vec>::from(artFeats);

         vec zta = reshape(articleFeat*user.t(),36,1,0);

        ARel = ARel + zta*zta.t();
        bRel = bRel + clicksVec[pos]*(1/posWeight[pos])*zta;

    }

    AInvRel = pinv(ARel);

}

void updateBucb(vector<long int>& selected, vector<double>& showsVec, vector<double>& clicksVec, struct line* line1, int flag)
{
    vector<unsigned int> posVec;
    for(vector<long int>::iterator it = selected.begin(); it != selected.end(); it++)
    {
        unsigned int currPos = std::find(gpOrder.begin(), gpOrder.end(), *it) - gpOrder.begin();
        posVec.push_back(currPos);
    }

    uvec pos = conv_to<uvec>::from(posVec);
    vec shows = conv_to<vec>::from(showsVec);
    vec clicks = conv_to<vec>::from(clicksVec);
    vec booster = conv_to<vec>::from(posWeight);
    booster = 1 / booster;
    booster = booster.subvec(0, batchSize-1);
    clicks = clicks % booster;

    unsigned int clusterId = line1->clusterId;

    uvec id = zeros<uvec>(1);
    id = clusterId;

    numPullsBucb(pos,id) = numPullsBucb(pos,id) + shows;
    fbBucb(pos,id) = ((fbBucb(pos,id) % (numPullsBucb(pos,id) - shows)) + clicks)/(numPullsBucb(pos,id));
    mat precBucb = diagmat(1/numPullsBucb.col(clusterId));

    if(flag == 1)
    {
        for(int i = 0; i < line1->numArts; i++)
        {
            artAgeBucb(indicesBucb(i)) = linesDone - articleBirth[line1->artId[i]];
        }

        mat cd = chol(kernel + precBucb * noiseVar);

        mat tempDiv1 = solve(cd, kernel.t());
        mat tempDiv2 = solve(cd.t(), tempDiv1);
        mat temp3 = kernel - kernel*tempDiv2;
        vec sigma2 = temp3.diag();

        sigma2 = abs(sigma2);

        mat tempDiv3 = solve(cd,fbBucb.col(clusterId));
        mat tempDiv4 = solve(cd.t(),tempDiv3);
        meansBucb.col(clusterId) = kernel * tempDiv4;

        bucbScore.col(clusterId) = meansBucb.col(clusterId) + beta % sqrt(sigma2);
    }
    else
    {
        mat kernel1 = kernel(indicesBucb, indicesBucb);
        mat prec1 = precBucb(indicesBucb, indicesBucb);

        vec fb1 = fbBucb(indicesBucb,id);
        vec beta1 = beta.elem(indicesBucb);

        mat cd = chol(kernel1+prec1*noiseVar);

        mat tempDiv1 = solve(cd, kernel1.t());
        mat tempDiv2 = solve(cd.t(), tempDiv1);
        mat temp3 = kernel1 - kernel1 * tempDiv2;
        vec sigma2 = temp3.diag();
        sigma2 = abs(sigma2);

        mat tempDiv3 = solve(cd,fb1);
        mat tempDiv4 = solve(cd.t(),tempDiv3);

        meansBucb(indicesBucb,id) = kernel1*tempDiv4;
        bucbScore(indicesBucb,id) = meansBucb(indicesBucb,id) + beta1 % sqrt(sigma2);
    }

    vector<long int>::iterator idIter;
    int bucbScoreIter;
    for(idIter = gpOrder.begin(),bucbScoreIter=0;idIter!=gpOrder.end();idIter++,bucbScoreIter++)
    {
        currentBucb[*idIter][clusterId] = bucbScore(bucbScoreIter,clusterId);
    }

}
    
void updateGp(vector<long int>& selected, vector<double>& showsVec, vector<double>& clicksVec, struct line* line1, int flag)
{
    vector<unsigned int> posVec;
    for(vector<long int>::iterator it = selected.begin(); it != selected.end(); it++)
    {
        unsigned int currPos = std::find(gpOrder.begin(), gpOrder.end(), *it) - gpOrder.begin();
        posVec.push_back(currPos);
    }

    uvec pos = conv_to<uvec>::from(posVec);
    vec shows = conv_to<vec>::from(showsVec);
    vec clicks = conv_to<vec>::from(clicksVec);
    vec booster = conv_to<vec>::from(posWeight);
    booster = 1 / booster;
    booster = booster.subvec(0, batchSize-1);
    clicks = clicks % booster;

    unsigned int clusterId = line1->clusterId;

    uvec id = zeros<uvec>(1);
    id = clusterId;

    numPulls(pos, id) = numPulls(pos,id) + shows;
    fb(pos, id) = ((fb(pos,id) % (numPulls(pos,id) - shows)) + clicks) / numPulls(pos,id);
    //prec precision metrix 
    mat prec = diagmat(1 / numPulls.col(clusterId));
    artAge = artAge+ones<vec>(numElem);

    beta = setBeta(artAge);

    if(flag == 1)
    {
        vector<unsigned int> positionVector;
        for(int i=0;i<line1->numArts;i++)
        {
            unsigned int artPos = std::find(gpOrder.begin(), gpOrder.end(), line1->artId[i]) - gpOrder.begin();
            positionVector.push_back(artPos);
            artAge(artPos) = linesDone - articleBirth[line1->artId[i]];

        }
        beta = setBeta(artAge);

        indices = conv_to<uvec>::from(positionVector);

        mat cd = chol(kernel + prec*noiseVar);

        mat tempDiv1 = solve(cd, kernel.t());
        mat tempDiv2 = solve(cd.t(), tempDiv1);
        mat temp3 = kernel - kernel * tempDiv2;
        vec sigma2 = temp3.diag();
        sigma2 = abs(sigma2);

        mat tempDiv3 = solve(cd, fb.col(clusterId));
        mat tempDiv4 = solve(cd.t(), tempDiv3);
        means.col(clusterId) = kernel * tempDiv4;

        sigma2 = abs(sigma2);

        ucbScore.col(clusterId) = means.col(clusterId)+beta%sqrt(sigma2);
    }
    else
    {
        mat kernel1 = kernel(indices, indices);
        mat prec1 = prec(indices, indices);

        vec fb1 = fb(indices, id);
        vec beta1 = beta.elem(indices);

        mat cd = chol(kernel1 + prec1 * noiseVar);

        mat tempDiv1 = solve(cd,kernel1.t());
        mat tempDiv2 = solve(cd.t(),tempDiv1);
        mat temp3 = kernel1 - kernel1*tempDiv2;
        vec sigma2 = temp3.diag();
        sigma2 = abs(sigma2);

        mat tempDiv3 = solve(cd,fb1);
        mat tempDiv4 = solve(cd.t(),tempDiv3);
        means(indices,id) = kernel1*tempDiv4;

        ucbScore(indices,id) = means(indices,id) + beta1 % sqrt(sigma2);

    }


    vector<long int>::iterator idIter;
    int ucbScoreIter;
    for(idIter = gpOrder.begin(),ucbScoreIter=0;idIter!=gpOrder.end();idIter++,ucbScoreIter++)
    {
        currentGp[*idIter][clusterId] = ucbScore(ucbScoreIter,clusterId);
    }

}

void simulator();

void buildKernel(double param)
{
    mat feats;
    feats.load("features.dat");
    int numArts = feats.n_rows;

    mat kernel2 = zeros<mat>(numArts, numArts);
    mat helper2 = zeros<mat>(numArts, numArts);
    mat Laplacian = zeros<mat>(numArts, numArts);
    colvec wtDeg = zeros<vec>(numArts);

    for(int i = 0; i < numArts; i++)
    {
        for(int j = i; j < numArts; j++)
        {
            helper2(i,j) = norm(feats.row(i)-feats.row(j), 2);
        }
    }
    helper2 = helper2 + helper2.t();

    wtDeg = sum(helper2, 1);
    Laplacian = diagmat(1./sqrt(wtDeg))*(diagmat(wtDeg) - helper2)*diagmat(1./sqrt(wtDeg));

    mat U ;
    mat V ;
    vec d;

    svd( U, d, V, Laplacian);

    vec d1 = diagvec(exp(param * diagmat(d)));
    kernel2 = U * diagmat(d1) * trans(V);

    kernel2= .5 * (kernel2 + kernel2.t());
    kernel = kernel2;
}

double addRandomNoise(double ctr, int pos)
{
    double click;
    double toss = rand() / double(RAND_MAX);
    if(toss < ctr * posWeight[pos])
        click = 1;
    else
        click =0;

    return click;
}

int main(int argc, char* argv[])
{

    if(argc!=4)
    {
        cout<<"usage:"<<argv[0]<<" kernel_param( alpha for diff) batch_size betaType"<<endl;
        return -1;
    }


    double param;

    buildKernel(atof(argv[1]));
    initialize();
    param = atof(argv[1]);

    batchSize = atoi(argv[2]);
    int betaType = atoi(argv[3]);
    struct stat st = {0};

    string baseString = "results_contextual_actual";
    ostringstream dirName;
    dirName << baseString << "alpha_" << param << "batch_" << batchSize << "_betaType" << betaType;

    if (stat(dirName.str().c_str(), &st) == -1) {
        mkdir(dirName.str().c_str(), 0700);
    }

    ostringstream idealFile;
    ostringstream ucbFile;
    ostringstream gpFile;
    ostringstream bucbFile;
    ostringstream randomFile;
    ostringstream linUcbFile;
    ostringstream linRelUcbFile;

    idealFile       << dirName.str() << "/" << "ideal.out";
    ucbFile         << dirName.str() << "/" << "ucb.out";
    gpFile          << dirName.str() << "/" << "gp.out";
    bucbFile        << dirName.str() << "/" << "bucb.out";
    randomFile      << dirName.str() << "/" << "random.out";
    linUcbFile      << dirName.str() << "/" << "linUcb.out";
    linRelUcbFile   << dirName.str() << "/" << "linRelUcb.out";

    simulator();

    dumpVector(idealStats,idealFile.str());
    dumpVector(ucbStats,ucbFile.str());
    dumpVector(gpStats,gpFile.str());
    dumpVector(bucbStats,bucbFile.str());
    dumpVector(randomStats,randomFile.str());
    dumpVector(linUcbStats, linUcbFile.str());
    dumpVector(linRelUcbStats, linRelUcbFile.str());

    return 0;
}


void simulator()
{
        vector<long int> idealSelected;
        vector<long int> ucbSelected;
        vector<long int> gpSelected;
        vector<long int> bucbSelected;
        vector<long int> randomOrder;
        vector<long int> randomSelected;
        vector<long int> linUcbSelected;
        vector<long int> linRelUcbSelected;

        int numFiles = 1;
        int flag = 0;
        int flagGP =0;
        int flagBUCB=0;

        long numArtsTotal;

        for(int fileNum = 2; fileNum<=10;fileNum++)
        {
            //string currFileName;
            string baseString = "dataset/ydata-fp-td-clicks-v1_0.200905";
            ostringstream currFileName;
            if(fileNum != 10)
            {
                currFileName << baseString << '0' << fileNum;
            }
            else
            {
                currFileName << baseString << fileNum;
            }

            ifstream infile;
            infile.open(currFileName.str().c_str(),fstream::in);
            cout << currFileName.str() << endl;
            string logLine;
            vector<struct ucbArt*>::iterator ucbIter;
            int numLines=0;
            while (getline(infile, logLine))
            {
                linesDone++;
                numShowsTotal++;
                struct line* line1 = new struct line();
                extractFromLogLine(line1,logLine);
                if(line1->click == 1)
                    numClicksTotal++;
                // initialize articles if a new one is observed
                for(int artIter = 0 ; artIter < line1->numArts ; artIter++)
                {
                    if(artHash1.find(line1->artId[artIter]) == artHash1.end())
                    {
                        struct artFeat* artFeat1 = new artFeat();
                        artFeat1->artId = line1->artId[artIter];
                        for(int iter = 0; iter < 6; iter++)
                        {
                            artFeat1->feats[iter] = ((struct artFeat*)(line1->artFeatList[artIter]))->feats[iter];
                        }

                        artHash1[line1->artId[artIter]] = artFeat1;
                        articleBirth[line1->artId[artIter]] = linesDone;
                        numArtsTotal++;
                        flag = 1;
                        flagGP=1;
                        flagBUCB=1;
                    }
                }

                // Clean up the selected articles
                idealSelected.clear();
                vector<long int>().swap(idealSelected);
                ucbSelected.clear();
                vector<long int>().swap(ucbSelected);
                gpSelected.clear();
                vector<long int>().swap(gpSelected);
                bucbSelected.clear();
                vector<long int>().swap(bucbSelected);
                randomOrder.clear();
                vector<long int>().swap(randomOrder);
                randomSelected.clear();
                vector<long int>().swap(randomSelected);
                linUcbSelected.clear();
                vector<long int>().swap(linUcbSelected);
                linRelUcbSelected.clear();
                vector<long int>().swap(linRelUcbSelected);

                // Select for each one of the types of algos the recommendation
                idealSelect     (line1, idealSelected,flag);
                ucbSelect       (line1, ucbSelected);
                gpSelect        (line1, gpSelected);
                bucbSelect      (line1, bucbSelected,flagBUCB);
                linUcbSelect    (line1, linUcbSelected);
                linRelUcbSelect (line1, linRelUcbSelected);

                vector <long int>::iterator it;
                int pos;
                if(idealSelected[0] == line1->dispArt)
                {
                    numShowsIdeal += batchSize;
                    for(pos=0,it = idealSelected.begin();it!=idealSelected.end();it++,pos++)
                    {
                        (((struct artFeat *)(artHash1[(long int) *it]))->showsIdeal)++;
                        double tempClicks = 1?(line1->click==1):0;
                        numClicksIdeal+=tempClicks;
                        (((struct artFeat *)(artHash1[(long int) *it]))->clicksIdeal)+=(tempClicks/posWeight[pos]);

                    }
                }

                if(ucbSelected[0] == line1->dispArt)
                {
                    totalShowsUCB += batchSize;
                    numShowsUCB[line1->clusterId]+=batchSize;
                    for(pos=0,it = ucbSelected.begin();it!=ucbSelected.end();it++,pos++)
                    {
                        (((struct artFeat *)(artHash1[(long int) *it]))->showsUCB[line1->clusterId])++;
                        double tempClicks = 1?(line1->click==1):0;
                        totalClicksUCB+=tempClicks;
                        numClicksUCB[line1->clusterId]+=tempClicks;
                        (((struct artFeat *)(artHash1[(long int) *it]))->clicksUCB[line1->clusterId])+=(tempClicks/posWeight[pos]);
                    }

                    for(int artIter = 0 ; artIter<line1->numArts ; artIter++)
                    {
                        long int currArtId = line1->artId[artIter];
                        if(((struct artFeat*)(artHash1[currArtId]))->showsUCB>0)
                        {
                            double tempScore;
                            tempScore = ((double) (((struct artFeat*)(artHash1[currArtId]))->clicksUCB[line1->clusterId])/(double) (((struct artFeat*)(artHash1[currArtId]))->showsUCB[line1->clusterId])) +
                                0.5*sqrt(2*log(double(numShowsUCB[line1->clusterId]))/(double) (((struct artFeat*)(artHash1[currArtId]))->showsUCB[line1->clusterId]));
                            currentUcb[currArtId][line1->clusterId] = tempScore;
                        }
                    }
                }

                if(gpSelected[0] == line1->dispArt)
                {
                    numShowsGP+=batchSize;

                    vector<double> lineShowsGP;
                    vector<double> lineClicksGP;

                    for(it = gpSelected.begin(),pos=0;it!=gpSelected.end();pos++,it++)
                    {
                        (((struct artFeat *)(artHash1[(long int)*it]))->showsGP)++;
                            //double clickIncrement = updateFeedback(idealSelected,line1->dispArt);
                        double tempClicks = line1->click;
                        numClicksGP += tempClicks;
                        (((struct artFeat *)(artHash1[(long int)*it]))->clicksGP)+=(tempClicks/posWeight[pos]);
                        lineShowsGP.push_back(1.0);
                        lineClicksGP.push_back(tempClicks);
                    }

                    updateGp(gpSelected, lineShowsGP,lineClicksGP, line1,flagGP);
                    flagGP=0;
                }

                if(bucbSelected[0]==line1->dispArt)
                {

                    numShowsBucb+=batchSize;

                    vector<double> lineShowsBucb;
                    vector<double> lineClicksBucb;

                    for(it = bucbSelected.begin(),pos=0;it!=bucbSelected.end();pos++,it++)
                    {
                        (((struct artFeat *)(artHash1[(long int)*it]))->showsBucb)++;
                            //double clickIncrement = updateFeedback(idealSelected,line1->dispArt);
                        double tempClicks = 1?(line1->click==1):0;
                        numClicksBucb+=tempClicks;
                        (((struct artFeat *)(artHash1[(long int)*it]))->clicksBucb)+=(tempClicks/posWeight[pos]);
                        lineShowsBucb.push_back(1.0);
                        lineClicksBucb.push_back(tempClicks);

                    }

                    updateBucb(bucbSelected, lineShowsBucb,lineClicksBucb, line1,flagBUCB);
                    flagBUCB=0;
                }

                if(linUcbSelected[0] == line1->dispArt)
                {
                    numShowsLinUcb+=batchSize;

                    vector<double> lineShowsLinUcb;
                    vector<double> lineClicksLinUcb;

                    for(it = linUcbSelected.begin(),pos=0;it!=linUcbSelected.end();pos++,it++)
                    {
                        (((struct artFeat *)(artHash1[(long int)*it]))->showsLinUcb)++;
                            //double clickIncrement = updateFeedback(idealSelected,line1->dispArt);
                        double tempClicks = 1?(line1->click==1):0;
                        numClicksLinUcb+=tempClicks;
                        (((struct artFeat *)(artHash1[(long int)*it]))->clicksLinUcb)+=(tempClicks/posWeight[pos]);
                        lineShowsLinUcb.push_back(1.0);
                        lineClicksLinUcb.push_back(tempClicks);

                    }

                    updateLinUcb(gpSelected, lineShowsLinUcb,lineClicksLinUcb, line1);
                }

                if(linRelUcbSelected[0] == line1->dispArt)
                {
                    numShowsLinRelUcb+=batchSize;

                    vector<double> lineShowsLinRelUcb;
                    vector<double> lineClicksLinRelUcb;

                    for(it = linRelUcbSelected.begin(),pos=0;it!=linRelUcbSelected.end();pos++,it++)
                    {
                        (((struct artFeat *)(artHash1[(long int)*it]))->showsLinRelUcb)++;
                            //double clickIncrement = updateFeedback(idealSelected,line1->dispArt);
                        double tempClicks = 1?(line1->click==1):0;
                        numClicksLinRelUcb+=tempClicks;
                        (((struct artFeat *)(artHash1[(long int)*it]))->clicksLinRelUcb)+=(tempClicks/posWeight[pos]);
                        lineShowsLinRelUcb.push_back(1.0);
                        lineClicksLinRelUcb.push_back(tempClicks);

                    }

                    updateLinRelUcb(gpSelected, lineShowsLinRelUcb,lineClicksLinRelUcb, line1);
                }


                for(int i=0; i<line1->numArts;i++)
                {
                    randomOrder.push_back(line1->artId[0]);
                    std::random_shuffle ( randomOrder.begin(), randomOrder.end() );

                }

                for(int i=0;i<batchSize;i++)
                {
                    randomSelected.push_back(randomOrder[i]);
                }

                if(randomSelected[0]==line1->dispArt)
                {
                    numShowsRandom+=batchSize;
                    for(it = randomSelected.begin(),pos=0;it!=randomSelected.end();pos++,it++)
                    {
                                //double clickIncrement = updateFeedback(idealSelected,line1->dispArt);
                        double tempClicks = 1?(line1->click==1):0;;
                        numClicksRandom+=tempClicks;
                    }
                }


                ((struct artFeat *)(artHash1[line1->dispArt]))->artShows++;
                if(line1->click == 1)
                    ((struct artFeat *)(artHash1[line1->dispArt]))->artClicks++;
                    if(((struct artFeat *)(artHash1[line1->dispArt]))->artClicks > 0)
                        ((struct artFeat *)artHash1[line1->dispArt])->ctr = (double)(((struct artFeat *)artHash1[line1->dispArt])->artClicks)/(double)(((struct artFeat *)artHash1[line1->dispArt])->artShows);



                if(linesDone%10000 == 0)
                {
                     struct outStatsLine* idealLine = new struct outStatsLine();
                    struct outStatsLine* ucbLine = new struct outStatsLine();
                    struct outStatsLine* gpLine = new struct outStatsLine();
                    struct outStatsLine* bucbLine = new struct outStatsLine();
                    struct outStatsLine* randomLine = new struct outStatsLine();
                    struct outStatsLine* linUcbLine = new struct outStatsLine();
                    struct outStatsLine* linRelUcbLine = new struct outStatsLine();

                    idealLine->shows = numShowsIdeal;
                    idealLine->clicks = numClicksIdeal;
                    idealLine->ctr = (double)numClicksIdeal/numShowsIdeal;

                    ucbLine->shows = totalShowsUCB;
                    ucbLine->clicks = totalClicksUCB;
                    ucbLine->ctr = (double)totalClicksUCB/totalShowsUCB;

                    gpLine->shows = numShowsGP;
                    gpLine->clicks = numClicksGP;
                    gpLine->ctr = (double)numClicksGP/numShowsGP;

                    bucbLine->shows = numShowsBucb;
                    bucbLine->clicks = numClicksBucb;
                    bucbLine->ctr = (double)numClicksBucb/numShowsBucb;

                    randomLine->shows = numShowsRandom;
                    randomLine->clicks = numClicksRandom;
                    randomLine->ctr = (double)numClicksRandom/numShowsRandom;

                    linUcbLine->shows = numShowsLinUcb;
                    linUcbLine->clicks = numClicksLinUcb;
                    linUcbLine->ctr = (double)numClicksLinUcb/numShowsLinUcb;

                    linRelUcbLine->shows = numShowsLinRelUcb;
                    linRelUcbLine->clicks = numClicksLinRelUcb;
                    linRelUcbLine->ctr = (double)numClicksLinRelUcb/numShowsLinRelUcb;

                    idealStats.push_back(idealLine);
                    ucbStats.push_back(ucbLine);
                    gpStats.push_back(gpLine);
                    bucbStats.push_back(bucbLine);
                    randomStats.push_back(randomLine);
                    linUcbStats.push_back(linUcbLine);
                    linRelUcbStats.push_back(linRelUcbLine);
                }

                if(linesDone%100000==0)
                {
                    cout<<linesDone<<endl;
                    cout<<"Ideal: "<<numShowsIdeal<<"-"<<numClicksIdeal<<"-"<<((double)numClicksIdeal)/numShowsIdeal<<endl;
                    cout<<"UCB: "<<totalShowsUCB<<"-"<<totalClicksUCB<<"-"<<((double)totalClicksUCB)/totalShowsUCB<<endl;
                    cout<<"GP: "<<numShowsGP<<"-"<<numClicksGP<<"-"<<((double)numClicksGP)/numShowsGP<<endl;
                    cout<<"BUCB: "<<numShowsBucb<<"-"<<numClicksBucb<<"-"<<((double)numClicksBucb)/numShowsBucb<<endl;
                    cout<<"Random: "<<numShowsRandom<<"-"<<numClicksRandom<<"-"<<((double)numClicksRandom)/numShowsRandom<<endl;
                    cout<<"LinUCB: "<<numShowsLinUcb<<"-"<<numClicksLinUcb<<"-"<<((double)numClicksLinUcb)/numShowsLinUcb<<endl;
                    cout<<"LinRelUCB: "<<numShowsLinRelUcb<<"-"<<numClicksLinRelUcb<<"-"<<((double)numClicksLinRelUcb)/numShowsLinRelUcb<<endl;

                }

                delete line1;
           // }//if(linesDones)

            }
            infile.close();


            cout<<numFiles<<" files done"<<endl;
            numFiles++;
                            //int tempChar;
                            //cin>>tempChar;
        }


}



