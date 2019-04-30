#include <opencv2/opencv.hpp>

#define SS 0
#define SE 1
#define ES 2
#define EE 3

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

enum GradientOperator { PREWITT_OPERATOR = 101, SOBEL_OPERATOR = 102, SCHARR_OPERATOR = 103 };

struct StackNode {
	int r, c;   // начальный пиксел
	int parent; // родитель (-1 если нету)
	int dir;    // направление
};

// для соединения граней
struct Chain {

	int dir;                   // направление цепочки
	int len;                   // кол-во пикселей в цепи
	int parent;                // родитель (-1 если нету)
	int children[2];           // дети (-1 если нету)
	cv::Point *pixels;         // указатель на начало массива пикселей
};




struct LS {
    cv::Point2d start;
    cv::Point2d end;
    
    LS(cv::Point2d _start, cv::Point2d _end)
    {
        start = _start;
        end = _end;
    }
};


struct LineSegment {
    double a, b;     
    int invert;
    
    double sx, sy;        // начало
    double ex, ey;        // конец
    
    int segmentNo;        // сегмент которому принадлежит отрезок
    int firstPixelIndex;  // индекс первого пикселя в сегменте
    int len;              // длина в пикселях
    
    LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len) {
        a = _a;
        b = _b;
        invert = _invert;
        sx = _sx;
        sy = _sy;
        ex = _ex;
        ey = _ey;
        segmentNo = _segmentNo;
        firstPixelIndex = _firstPixelIndex;
        len = _len;
    }
};

class EDLines {
							
public:
	EDLines(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.5, bool _sumFlag = true, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);

	cv::Mat getEdgeImage();
	cv::Mat getAnchorImage();
	cv::Mat getSmoothImage();
	cv::Mat getGradImage();
    cv::Mat getLineImage();
    cv::Mat drawOnImage();
	
	int getSegmentNo();
	int getAnchorNo();
	
	std::vector<cv::Point> getAnchorPoints();
	std::vector<std::vector<cv::Point>> getSegments();
	std::vector<std::vector<cv::Point>> getSortedSegments();
	
	cv::Mat drawParticularSegments(std::vector<int> list);
    
    std::vector<LS> getLines();
    int getLinesNo();

protected:
	int width; // ширина исходного изображения
	int height; // высота исходного изображения
	uchar *srcImg; 
	std::vector<std::vector< cv::Point> > segmentPoints;
	double sigma; // сигма Гаусса
	cv::Mat smoothImage;
	uchar *edgeImg;
	uchar *smoothImg; 
	int segmentNos;
	int minPathLen;
	cv::Mat srcImage;

private:
	void ComputeGradient();
	void ComputeAnchorPoints();
	void JoinAnchorPointsUsingSortedAnchors(); 
	int* sortAnchorsByGradValue1();

	static int LongestChain(Chain *chains, int root);
	static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

	int anchorNos;
	std::vector<cv::Point> anchorPoints;
	std::vector<cv::Point> edgePoints;

	cv::Mat edgeImage;
	cv::Mat gradImage;
    cv::Mat threshImage;

	uchar *dirImg; // указатель на направление градиента пикселя
	short *gradImg; // указатель на градиет пикселя

	GradientOperator gradOperator; // оператор градиента
	int gradThresh;
	int anchorThresh;
	int scanInterval;
	bool sumFlag;
    
    std::vector<LineSegment> lines;
    std::vector<LineSegment> invalidLines;
    std::vector<LS> linePoints;
    int linesNo;
    int min_line_len;
    double line_error;
    double max_distance_between_two_lines;
    double max_error;
    double prec;
    
    
    int ComputeMinLineLength();
    void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);
    void JoinCollinearLines();
    
    bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);
    
    static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
    static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
    static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
    static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
    static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
    static void UpdateLineParameters(LineSegment *ls);
    static void EnumerateRectPoints(double sx, double sy, double ex, double ey,int ptsx[], int ptsy[], int *pNoPoints);

};


