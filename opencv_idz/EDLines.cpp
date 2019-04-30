#include "EDLines.h"


using namespace cv;
using namespace std;

EDLines::EDLines(Mat _srcImage, GradientOperator _op, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag, double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
{	
	// Проверка параметров на адекватность
	if (_gradThresh < 1) _gradThresh = 1;
	if (_anchorThresh < 0) _anchorThresh = 0;
	if (_sigma < 1.0) _sigma = 1.0;

	srcImage = _srcImage;

	height = srcImage.rows;
	width = srcImage.cols;
	
	gradOperator = _op;
	gradThresh = _gradThresh;
	anchorThresh = _anchorThresh;
	scanInterval = _scanInterval;
	minPathLen = _minPathLen;
	sigma = _sigma;
	sumFlag = _sumFlag;

	segmentNos = 0;
	segmentPoints.push_back(vector<Point>()); // создаем пустой вектор точек для сегментов

	edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); // инициализируем изображение с гранями
	smoothImage = Mat(height, width, CV_8UC1);
	gradImage = Mat(height, width, CV_16SC1);
    
    smoothImg = smoothImage.data;
    gradImg = (short*)gradImage.data;
    edgeImg = edgeImage.data;

	srcImg = srcImage.data;
    dirImg = new unsigned char[width*height];
	
	/*------------ ПРИМЕНЯЕМ РАЗМЫТИЕ ПО ГАУССУ -------------------*/
	if (sigma == 1.0)
		GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
	else
		GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma

	/*------------ ВЫСЧИТЫВАЕМ ГРАДИЕНТ И НАПРАВЛЕНИЕ ГРАНЕЙ -------------------*/
	ComputeGradient();

	/*------------ ВЫСЧИТЫВАЕМ ОПОРНЫЕ ТОЧКИ -------------------*/
	ComputeAnchorPoints();

	/*------------ СОЕДИНЯЕМ ОПОРНЫЕ ТОЧКИ -------------------*/
	JoinAnchorPointsUsingSortedAnchors();
	
	delete[] dirImg;
    
    min_line_len = _min_line_len;
    line_error = _line_error;
    max_distance_between_two_lines = _max_distance_between_two_lines;
    max_error = _max_error;
    
    if(min_line_len == -1) // если не задано значение, посчитаем
        min_line_len = ComputeMinLineLength();
    
    if (min_line_len < 9) // устанавливаем мин длину отрезка
        min_line_len = 9;
    
    
    
    double *x = new double[(width+height) * 8];
    double *y = new double[(width+height) * 8];
    
    linesNo = 0;
    
    // Обрабатывем каждый сегмент
    for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
        std::vector<Point> segment = segmentPoints[segmentNumber];
        for (int k = 0; k < segment.size(); k++) {
            x[k] = segment[k].x;
            y[k] = segment[k].y;
        }
        SplitSegment2Lines(x, y, segment.size(), segmentNumber);
    }
    
    /*----------- СОЕДИНЯЕМ КОЛЛИНЕАРНЫЕ ОТРЕЗКИ----------------*/
    JoinCollinearLines();
    
    for (int i = 0; i<linesNo; i++) {
        Point2d start(lines[i].sx, lines[i].sy);
        Point2d end(lines[i].ex, lines[i].ey);
        
        linePoints.push_back(LS(start, end));
    }
    
    delete[] x;
    delete[] y;
}


Mat EDLines::getEdgeImage()
{
	return edgeImage;
}

Mat EDLines::getAnchorImage()
{
	Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

	std::vector<Point>::iterator it;

	for (it = anchorPoints.begin(); it != anchorPoints.end(); it++)
		anchorImage.at<uchar>(*it) = 255;

	return anchorImage;
}

Mat EDLines::getSmoothImage()
{
	return smoothImage;
}

Mat EDLines::getGradImage()
{	
	Mat result8UC1;
	convertScaleAbs(gradImage, result8UC1);
	
	return result8UC1;
}


int EDLines::getSegmentNo()
{
	return segmentNos;
}

int EDLines::getAnchorNo()
{
	return anchorNos;
}

std::vector<Point> EDLines::getAnchorPoints()
{
	return anchorPoints;
}

std::vector<std::vector<Point>> EDLines::getSegments()
{
	return segmentPoints;
}

std::vector<std::vector<Point>> EDLines::getSortedSegments()
{
		// сортируем сегметы по убыванию длины
		std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point> & a, const std::vector<Point> & b) { return a.size() > b.size(); });

		return segmentPoints;
}

Mat EDLines::drawParticularSegments(std::vector<int> list)
{
	Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

	std::vector<Point>::iterator it;
	std::vector<int>::iterator itInt;

	for (itInt = list.begin(); itInt != list.end(); itInt++)
		for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
			segmentsImage.at<uchar>(*it) = 255;
	
	return segmentsImage;
}


void EDLines::ComputeGradient()
{	
	// инициализируем градиент
	for (int j = 0; j<width; j++) { gradImg[j] = gradImg[(height - 1)*width + j] = gradThresh - 1; }
	for (int i = 1; i<height - 1; i++) { gradImg[i*width] = gradImg[(i + 1)*width - 1] = gradThresh - 1; }

	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {

			int com1 = smoothImg[(i + 1)*width + j + 1] - smoothImg[(i - 1)*width + j - 1];
			int com2 = smoothImg[(i - 1)*width + j + 1] - smoothImg[(i + 1)*width + j - 1];

			int gx;
			int gy;
			
			switch (gradOperator)
			{
			case PREWITT_OPERATOR:
				gx = abs(com1 + com2 + (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(com1 - com2 + (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
				break;
			case SOBEL_OPERATOR:
				gx = abs(com1 + com2 + 2 * (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(com1 - com2 + 2 * (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
				break;
			case SCHARR_OPERATOR:
				gx = abs(3 * (com1 + com2) + 10 * (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(3 * (com1 - com2) + 10 * (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
			}
			
			int sum;

			if(sumFlag)
				sum = gx + gy;
			else
				sum = (int)sqrt((double)gx*gx + gy*gy);

			int index = i*width + j;
			gradImg[index] = sum;

			if (sum >= gradThresh) {
				if (gx >= gy) dirImg[index] = EDGE_VERTICAL;
				else          dirImg[index] = EDGE_HORIZONTAL;
			}
		}
	}
}

void EDLines::ComputeAnchorPoints()
{
	for (int i = 2; i<height - 2; i++) {
		int start = 2;
		int inc = 1;
		if (i%scanInterval != 0) { start = scanInterval; inc = scanInterval; }

		for (int j = start; j<width - 2; j += inc) {
			if (gradImg[i*width + j] < gradThresh) continue;

			if (dirImg[i*width + j] == EDGE_VERTICAL) {
				// vertical edge
				int diff1 = gradImg[i*width + j] - gradImg[i*width + j - 1];
				int diff2 = gradImg[i*width + j] - gradImg[i*width + j + 1];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}

			}
			else {
				// horizontal edge
				int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j];
				int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}
			}
		}
	}

	anchorNos = anchorPoints.size(); // суммарное число опорных точек
}

void EDLines::JoinAnchorPointsUsingSortedAnchors()
{
	int *chainNos = new int[(width + height) * 8];

	Point *pixels = new Point[width*height];
	StackNode *stack = new StackNode[width*height];
	Chain *chains = new Chain[width*height];

	// сортируем опорные точки по убыванию градиента в них
	int *A = sortAnchorsByGradValue1();

	// соединяем опорные точки начиная с наибольших значений градиента
	int totalPixels = 0;

	for (int k = anchorNos - 1; k >= 0; k--) {
		int pixelOffset = A[k];

		int i = pixelOffset / width;
		int j = pixelOffset % width;


		if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

		chains[0].len = 0;
		chains[0].parent = -1;
		chains[0].dir = 0;
		chains[0].children[0] = chains[0].children[1] = -1;
		chains[0].pixels = NULL;


		int noChains = 1;
		int len = 0;
		int duplicatePixelCount = 0;
		int top = -1;  // вершина стека

		if (dirImg[i*width + j] == EDGE_VERTICAL) {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = DOWN;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = UP;
			stack[top].parent = 0;

		}
		else {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = RIGHT;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = LEFT;
			stack[top].parent = 0;
		} //end-else

		  // пока стек не пуст
	StartOfWhile:
		while (top >= 0) {
			int r = stack[top].r;
			int c = stack[top].c;
			int dir = stack[top].dir;
			int parent = stack[top].parent;
			top--;

			if (edgeImg[r*width + c] != EDGE_PIXEL) duplicatePixelCount++;

			chains[noChains].dir = dir;   // traversal direction
			chains[noChains].parent = parent;
			chains[noChains].children[0] = chains[noChains].children[1] = -1;


			int chainLen = 0;

			chains[noChains].pixels = &pixels[len];

			pixels[len].y = r;
			pixels[len].x = c;
			len++;
			chainLen++;

			if (dir == LEFT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань горизонтальная. Направлена влево
					//
					//   A
					//   B x 
					//   C 
					//
					// очищаем верхний и нижний пиксели
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[r*width + c - 1] >= ANCHOR_PIXEL) { c--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель СЛЕВА
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[r*width + c - 1];
						int C = gradImg[(r + 1)*width + c - 1];

						if (A > B) {
							if (A > C) r--;
							else       r++;
						}
						else  if (C > B) r++;
						c--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else if (dir == RIGHT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань горизонтальная. Направлена вправо
					//
					//     A
					//   x B
					//     C
					//
					// очищаем верхний и нижний пиксели
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[r*width + c + 1] >= ANCHOR_PIXEL) { c++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель СПРАВА
						int A = gradImg[(r - 1)*width + c + 1];
						int B = gradImg[r*width + c + 1];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) r--;       // A
							else       r++;       // C
						}
						else if (C > B) r++;  // C
						c++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;

			}
			else if (dir == UP) {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань вертикальная. Направлена вверх
					//
					//   A B C
					//     x
					//
					// очищаем левый и правый пиксели
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[(r - 1)*width + c] >= ANCHOR_PIXEL) { r--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель ВВЕРХ
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[(r - 1)*width + c];
						int C = gradImg[(r - 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;
							else       c++;
						}
						else if (C > B) c++;
						r--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// Грань вертикальная. Направлена вниз
					//
					//     x
					//   A B C
					//
					// очищаем пиксле слева и справа
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;

					// ищем пиксель на грани среди соседей
					if (edgeImg[(r + 1)*width + c] >= ANCHOR_PIXEL) { r++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// иначе -- идем в максимальный по градиенту пиксель ВНИЗУ
						int A = gradImg[(r + 1)*width + c - 1];
						int B = gradImg[(r + 1)*width + c];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;       // A
							else       c++;       // C
						}
						else if (C > B) c++;  // C
						r++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}

					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;
			}

		}


		if (len - duplicatePixelCount < minPathLen) {
			for (int k = 0; k<len; k++) {

				edgeImg[pixels[k].y*width + pixels[k].x] = 0;
				edgeImg[pixels[k].y*width + pixels[k].x] = 0;

			}

		}
		else {

			int noSegmentPixels = 0;

			int totalLen = LongestChain(chains, chains[0].children[1]);

			if (totalLen > 0) {
				int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

				// копируем пиксели в обратном порядке
				for (int k = count - 1; k >= 0; k--) {
					int chainNo = chainNos[k];

                    /* Пробуем удалить лишние пиксели */

                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0) {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1) {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else break;
                    } //end-while

                    if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1) chains[chainNo].len--;
                    }

					for (int l = chains[chainNo].len - 1; l >= 0; l--) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0;  // помечаем скопированной
				}
			}

			totalLen = LongestChain(chains, chains[0].children[0]);
			if (totalLen > 1) {
				
				int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);
                
                // копируем цепочку в прямом порядке. пропускаем первый пиксель в цепи
				int lastChainNo = chainNos[0];
				chains[lastChainNo].pixels++;
				chains[lastChainNo].len--;

				for (int k = 0; k<count; k++) {
					int chainNo = chainNos[k];

					/* Пробуем удалить лишние пиксели */
					int fr = chains[chainNo].pixels[0].y;
					int fc = chains[chainNo].pixels[0].x;

					int index = noSegmentPixels - 2;
					while (index >= 0) {
						int dr = abs(fr - segmentPoints[segmentNos][index].y);
						int dc = abs(fc - segmentPoints[segmentNos][index].x);

						if (dr <= 1 && dc <= 1) {
							segmentPoints[segmentNos].pop_back();
							noSegmentPixels--;
							index--;
						}
						else break;
					}

					int startIndex = 0;
					int chainLen = chains[chainNo].len;
					if (chainLen > 1 && noSegmentPixels > 0) {
						int fr = chains[chainNo].pixels[1].y;
						int fc = chains[chainNo].pixels[1].x;

						int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
						int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

						if (dr <= 1 && dc <= 1) { startIndex = 1; }
					}

					for (int l = startIndex; l<chains[chainNo].len; l++) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0;  // помечаем скопированной
				}
			}


			  //  Пробуем удалить лишние пиксели
			int fr = segmentPoints[segmentNos][1].y;
			int fc = segmentPoints[segmentNos][1].x;


			int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
			int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);


			if (dr <= 1 && dc <= 1) {
				segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
				noSegmentPixels--;
			} //end-if

			segmentNos++;
			segmentPoints.push_back(vector<Point>());

													  // копируем оставшиеся цепочки сюда
			for (int k = 2; k<noChains; k++) {
				if (chains[k].len < 2) continue;

				totalLen = LongestChain(chains, k);

				if (totalLen >= 10) {

					int count = RetrieveChainNos(chains, k, chainNos);

					// копируем пиксели
					noSegmentPixels = 0;
					for (int k = 0; k<count; k++) {
						int chainNo = chainNos[k];
					
						/* Пробуем удалить лишние пиксели */
						int fr = chains[chainNo].pixels[0].y;
						int fc = chains[chainNo].pixels[0].x;

						int index = noSegmentPixels - 2;
						while (index >= 0) {
							int dr = abs(fr - segmentPoints[segmentNos][index].y);
							int dc = abs(fc - segmentPoints[segmentNos][index].x);

							if (dr <= 1 && dc <= 1) {
								// удаляем последний пиксель т к соседи
								segmentPoints[segmentNos].pop_back();
								noSegmentPixels--;
								index--;
							}
							else break;
						}

						int startIndex = 0;
						int chainLen = chains[chainNo].len;
						if (chainLen > 1 && noSegmentPixels > 0) {
							int fr = chains[chainNo].pixels[1].y;
							int fc = chains[chainNo].pixels[1].x;

							int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
							int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

							if (dr <= 1 && dc <= 1) { startIndex = 1; }
						}
						for (int l = startIndex; l<chains[chainNo].len; l++) {
							segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
							noSegmentPixels++;
						}

						chains[chainNo].len = 0;  // помечаем скопироавнной
					}
					segmentPoints.push_back(vector<Point>());
					segmentNos++;
				}
			}

		}

	}

    // удаляем последний сегмент из массива, т.к. он пуст
	segmentPoints.pop_back();

	delete[] A;
	delete[] chains;
	delete[] stack;
	delete[] chainNos;
	delete[] pixels;
}

int * EDLines::sortAnchorsByGradValue1()
{
	int SIZE = 128 * 256;
	int *C = new int[SIZE];
	memset(C, 0, sizeof(int)*SIZE);

	// считаем кол-во значений градиента
	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

			int grad = gradImg[i*width + j];
			C[grad]++;
		}
	} //end-for 

	// считаем индексы
    for (int i = 1; i<SIZE; i++) C[i] += C[i - 1];

	int noAnchors = C[SIZE - 1];
	int *A = new int[noAnchors];
	memset(A, 0, sizeof(int)*noAnchors);


	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

			int grad = gradImg[i*width + j];
			int index = --C[grad];
			A[index] = i*width + j;    // сдвиг опорной точки
		}
	}

	delete[] C;

	return A;

}


int EDLines::LongestChain(Chain *chains, int root) {
	if (root == -1 || chains[root].len == 0) return 0;

	int len0 = 0;
	if (chains[root].children[0] != -1) len0 = LongestChain(chains, chains[root].children[0]);

	int len1 = 0;
	if (chains[root].children[1] != -1) len1 = LongestChain(chains, chains[root].children[1]);

	int max = 0;

	if (len0 >= len1) {
		max = len0;
		chains[root].children[1] = -1;

	}
	else {
		max = len1;
		chains[root].children[0] = -1;
	}

	return chains[root].len + max;
}

int EDLines::RetrieveChainNos(Chain * chains, int root, int chainNos[])
{
	int count = 0;

	while (root != -1) {
		chainNos[count] = root;
		count++;

		if (chains[root].children[0] != -1) root = chains[root].children[0];
		else                                root = chains[root].children[1];
	}

	return count;
}


vector<LS> EDLines::getLines()
{
    return linePoints;
}

int EDLines::getLinesNo()
{
    return linesNo;
}

Mat EDLines::getLineImage()
{
    Mat lineImage = Mat(height, width, CV_8UC1, Scalar(0));
    for (int i = 0; i < linesNo; i++) {
        line(lineImage, linePoints[i].start, linePoints[i].end, Scalar(255), 1, LINE_AA, 0);
    }
    
    return lineImage;
}

Mat EDLines::drawOnImage()
{
    Mat colorImage = Mat(height, width, CV_8UC1, srcImg);
    cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
    for (int i = 0; i < linesNo; i++) {
        line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
    }
    
    return colorImage;
}

// Считает минимальную длину линии используя формулу NFA
int EDLines::ComputeMinLineLength() {

    double logNT = 2.0*(log10((double)width) + log10((double)height));
    return (int) round((-logNT / log10(0.125))*0.5);
}

//-----------------------------------------------------------------
// разбивает цепочку отрезков на линии
//
void EDLines::SplitSegment2Lines(double * x, double * y, int noPixels, int segmentNo)
{
    
    // первый пиксель линии внутри сегмента
    int firstPixelIndex = 0;
    
    while (noPixels >= min_line_len) {
        // пытаемся создать линию минимальной длины
        bool valid = false;
        double lastA, lastB, error;
        int lastInvert;
        
        while (noPixels >= min_line_len) {
            LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
            if (error <= 0.5) { valid = true; break; }
            
            noPixels -= 1;
            x += 1; y += 1;
            firstPixelIndex += 1;
        }
        
        if (valid == false) return;
        
        // пытаемся удлинить линию
        int index = min_line_len;
        int len = min_line_len;
        
        while (index < noPixels) {
            int startIndex = index;
            int lastGoodIndex = index - 1;
            int goodPixelCount = 0;
            int badPixelCount = 0;
            while (index < noPixels) {
                double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);
                
                if (d <= line_error) {
                    lastGoodIndex = index;
                    goodPixelCount++;
                    badPixelCount = 0;
                    
                }
                else {
                    badPixelCount++;
                    if (badPixelCount >= 5) break;
                }
                
                index++;
            }
            
            if (goodPixelCount >= 2) {
                len += lastGoodIndex - startIndex + 1;
                LineFit(x, y, len, lastA, lastB, lastInvert);
                index = lastGoodIndex + 1;
            }
            
            if (goodPixelCount < 2 || index >= noPixels) {
                // завершаем линию, подсчитывем конечную точку
                double sx, sy, ex, ey;
                
                int index = 0;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
                int noSkippedPixels = index;
                
                index = lastGoodIndex;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);
                
                // добавляем линию в список
                lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
                linesNo++;
                len = index + 1;
                break;
            }
        }
        
        noPixels -= len;
        x += len;
        y += len;
        firstPixelIndex += len;
    }
}

//------------------------------------------------------------------
// Соединяем коллинеарные линии
//
void EDLines::JoinCollinearLines()
{
    int lastLineIndex = -1;
    int i = 0;
    while (i < linesNo) {
        int segmentNo = lines[i].segmentNo;
        
        lastLineIndex++;
        if (lastLineIndex != i)
            lines[lastLineIndex] = lines[i];
        
        int firstLineIndex = lastLineIndex;
        
        int count = 1;
        for (int j = i + 1; j< linesNo; j++) {
            if (lines[j].segmentNo != segmentNo) break;
            
            // пытаемся соединить текущую линию с предыдущей в сегменте
            if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j],
                                         lastLineIndex) == false) {
                lastLineIndex++;
                if (lastLineIndex != j)
                    lines[lastLineIndex] = lines[j];
                
            }
            
            count++;
        }
        
        // пытаемся замкнуть сегмент
        if (firstLineIndex != lastLineIndex) {
            if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex],
                                         firstLineIndex)) {
                lastLineIndex--;
            }
        }
        
        i += count;
    }
    
    linesNo = lastLineIndex + 1;
}

double EDLines::ComputeMinDistance(double x1, double y1, double a, double b, int invert)
{
    double x2, y2;
    
    if (invert == 0) {
        if (b == 0) {
            x2 = x1;
            y2 = a;
            
        }
        else {
            double d = -1.0 / (b);
            double c = y1 - d*x1;
            
            x2 = (a - c) / (d - b);
            y2 = a + b*x2;
        }
        
    }
    else {
        if (b == 0) {
            x2 = a;
            y2 = y1;
            
        }
        else {
            double d = -1.0 / (b);
            double c = x1 - d*y1;
            
            y2 = (a - c) / (d - b);
            x2 = a + b*y2;
        }
    }
    
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

//---------------------------------------------------------------------------------
// ищем близжайшие точки на линии
//
void EDLines::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut)
{
    double x2, y2;
    
    if (invert == 0) {
        if (b == 0) {
            x2 = x1;
            y2 = a;
            
        }
        else {
            double d = -1.0 / (b);
            double c = y1 - d*x1;
            
            x2 = (a - c) / (d - b);
            y2 = a + b*x2;
        }
        
    }
    else {
        if (b == 0) {
            x2 = a;
            y2 = y1;
            
        }
        else {
            double d = -1.0 / (b);
            double c = x1 - d*y1;
            
            y2 = (a - c) / (d - b);
            x2 = a + b*y2;
        }
    }
    
    xOut = x2;
    yOut = y2;
}

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
// Assumes that the direction of the line is known by a previous computation
//
void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, int invert)
{
    if (count<2) return;
    
    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i<count; i++) {
        Sx += x[i];
        Sy += y[i];
    } //end-for
    
    if (invert) {
        // Vertical line. Swap x & y, Sx & Sy
        double *t = x;
        x = y;
        y = t;
        
        double d = Sx;
        Sx = Sy;
        Sy = d;
    } //end-if
    
    // Now compute Sxx & Sxy
    for (int i = 0; i<count; i++) {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    } //end-for
    
    double D = S*Sxx - Sx*Sx;
    a = (Sxx*Sy - Sx*Sxy) / D;
    b = (S  *Sxy - Sx* Sy) / D;
}



void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, double &e, int &invert)
{
    if (count<2) return;
    
    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i<count; i++) {
        Sx += x[i];
        Sy += y[i];
    }
    
    double mx = Sx / count;
    double my = Sy / count;
    
    double dx = 0.0;
    double dy = 0.0;
    for (int i = 0; i < count; i++) {
        dx += (x[i] - mx)*(x[i] - mx);
        dy += (y[i] - my)*(y[i] - my);
    }
    
    if (dx < dy) {
        invert = 1;
        double *t = x;
        x = y;
        y = t;
        
        double d = Sx;
        Sx = Sy;
        Sy = d;
        
    }
    else {
        invert = 0;
    }

    for (int i = 0; i<count; i++) {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    }
    
    double D = S*Sxx - Sx*Sx;
    a = (Sxx*Sy - Sx*Sxy) / D;
    b = (S  *Sxy - Sx* Sy) / D;
    
    if (b == 0.0) {
        double error = 0.0;
        for (int i = 0; i<count; i++) {
            error += fabs((a) - y[i]);
        }
        e = error / count;
        
    }
    else {
        double error = 0.0;
        for (int i = 0; i<count; i++) {
            double d = -1.0 / (b);
            double c = y[i] - d*x[i];
            double x2 = ((a) - c) / (d - (b));
            double y2 = (a) + (b)*x2;
            
            double dist = (x[i] - x2)*(x[i] - x2) + (y[i] - y2)*(y[i] - y2);
            error += dist;
        }
        
        e = sqrt(error / count);
    }
}

//-----------------------------------------------------------------
// проверяет коллинеарность линий и соединяет
// при соединении ls1 обновляется, а ls2 не меняется
// возвращает true если линии объединены
//
bool EDLines::TryToJoinTwoLineSegments(LineSegment * ls1, LineSegment * ls2, int changeIndex)
{
    int which;
    double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
    if (dist > max_distance_between_two_lines) return false;
    
    // считаем длину линий. используем максимальную из них
    double dx = ls1->sx - ls1->ex;
    double dy = ls1->sy - ls1->ey;
    double prevLen = sqrt(dx*dx + dy*dy);
    
    dx = ls2->sx - ls2->ex;
    dy = ls2->sy - ls2->ey;
    double nextLen = sqrt(dx*dx + dy*dy);
    
    // используем максимальную
    LineSegment *shorter = ls1;
    LineSegment *longer = ls2;
    
    if (prevLen > nextLen) { shorter = ls2; longer = ls1; }
    
    // используем три точки для проверки коллинеарности
    dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);
    
    dist /= 3.0;
    
    if (dist > max_error) return false;
    

    /// 4 варианта: 1:(s1, s2), 2:(s1, e2), 3:(e1, s2), 4:(e1, e2)
    
    /// 1: (s1, s2)
    dx = fabs(ls1->sx - ls2->sx);
    dy = fabs(ls1->sy - ls2->sy);
    double d = dx + dy;
    double max = d;
    which = 1;
    
    /// 2: (s1, e2)
    dx = fabs(ls1->sx - ls2->ex);
    dy = fabs(ls1->sy - ls2->ey);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 2;
    }
    
    /// 3: (e1, s2)
    dx = fabs(ls1->ex - ls2->sx);
    dy = fabs(ls1->ey - ls2->sy);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 3;
    }
    
    /// 4: (e1, e2)
    dx = fabs(ls1->ex - ls2->ex);
    dy = fabs(ls1->ey - ls2->ey);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 4;
    }
    
    if (which == 1) {
        // (s1, s2)
        ls1->ex = ls2->sx;
        ls1->ey = ls2->sy;
        
    }
    else if (which == 2) {
        // (s1, e2)
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
        
    }
    else if (which == 3) {
        // (e1, s2)
        ls1->sx = ls2->sx;
        ls1->sy = ls2->sy;
        
    }
    else {
        // (e1, e2)
        ls1->sx = ls1->ex;
        ls1->sy = ls1->ey;
        
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    }
    

    
    // обновляем параметры первой линии
    if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex) ls1->len += ls2->len;
    else if (ls2->len > ls1->len) {
        ls1->firstPixelIndex = ls2->firstPixelIndex;
        ls1->len = ls2->len;
    }
    
    UpdateLineParameters(ls1);
    lines[changeIndex] = *ls1;
    
    return true;
}

//-------------------------------------------------------------------------------
// посчитываем минимальное расстояние между концами двух отрезков
//
double EDLines::ComputeMinDistanceBetweenTwoLines(LineSegment * ls1, LineSegment * ls2, int * pwhich)
{
    double dx = ls1->sx - ls2->sx;
    double dy = ls1->sy - ls2->sy;
    double d = sqrt(dx*dx + dy*dy);
    double min = d;
    int which = SS;
    
    dx = ls1->sx - ls2->ex;
    dy = ls1->sy - ls2->ey;
    d = sqrt(dx*dx + dy*dy);
    if (d < min) { min = d; which = SE; }
    
    dx = ls1->ex - ls2->sx;
    dy = ls1->ey - ls2->sy;
    d = sqrt(dx*dx + dy*dy);
    if (d < min) { min = d; which = ES; }
    
    dx = ls1->ex - ls2->ex;
    dy = ls1->ey - ls2->ey;
    d = sqrt(dx*dx + dy*dy);
    if (d < min) { min = d; which = EE; }
    
    if (pwhich) *pwhich = which;
    return min;
}


void EDLines::UpdateLineParameters(LineSegment * ls)
{
    double dx = ls->ex - ls->sx;
    double dy = ls->ey - ls->sy;
    
    if (fabs(dx) >= fabs(dy)) {
        ls->invert = 0;
        if (fabs(dy) < 1e-3) { ls->b = 0; ls->a = (ls->sy + ls->ey) / 2; }
        else {
            ls->b = dy / dx;
            ls->a = ls->sy - (ls->b)*ls->sx;
        }
        
    }
    else {
        ls->invert = 1;
        if (fabs(dx) < 1e-3) { ls->b = 0; ls->a = (ls->sx + ls->ex) / 2; }
        else {
            ls->b = dx / dy;
            ls->a = ls->sx - (ls->b)*ls->sy;
        }
    }
}

void EDLines::EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int * pNoPoints)
{
    double vxTmp[4], vyTmp[4];
    double vx[4], vy[4];
    int n, offset;
    
    double x1 = sx;
    double y1 = sy;
    double x2 = ex;
    double y2 = ey;
    double width = 2;
    
    double dx = x2 - x1;
    double dy = y2 - y1;
    double vLen = sqrt(dx*dx + dy*dy);
    
    dx = dx / vLen;
    dy = dy / vLen;
    
    vxTmp[0] = x1 - dy * width / 2.0;
    vyTmp[0] = y1 + dx * width / 2.0;
    vxTmp[1] = x2 - dy * width / 2.0;
    vyTmp[1] = y2 + dx * width / 2.0;
    vxTmp[2] = x2 + dy * width / 2.0;
    vyTmp[2] = y2 - dx * width / 2.0;
    vxTmp[3] = x1 + dy * width / 2.0;
    vyTmp[3] = y1 - dx * width / 2.0;
    
    if (x1 < x2 && y1 <= y2) offset = 0;
    else if (x1 >= x2 && y1 < y2) offset = 1;
    else if (x1 > x2 && y1 >= y2) offset = 2;
    else                          offset = 3;
    
    for (n = 0; n<4; n++) {
        vx[n] = vxTmp[(offset + n) % 4];
        vy[n] = vyTmp[(offset + n) % 4];
    }
    

    int x = (int)ceil(vx[0]) - 1;
    int y = (int)ceil(vy[0]);
    double ys = -DBL_MAX, ye = -DBL_MAX;
    
    int noPoints = 0;
    while (1) {
       
        y++;
        while (y > ye && x <= vx[2]) {
            x++;
            
            if (x > vx[2]) break;
            if ((double)x < vx[3]) {
                if (fabs(vx[0] - vx[3]) <= 0.01) {
                    if (vy[0]<vy[3]) ys = vy[0];
                    else if (vy[0]>vy[3]) ys = vy[3];
                    else     ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
                }
                else
                    ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
                
            }
            else {
                if (fabs(vx[3] - vx[2]) <= 0.01) {
                    if (vy[3]<vy[2]) ys = vy[3];
                    else if (vy[3]>vy[2]) ys = vy[2];
                    else     ys = vy[3] + (x - vx[3]) * (y2 - vy[3]) / (vx[2] - vx[3]);
                }
                else
                    ys = vy[3] + (x - vx[3]) * (vy[2] - vy[3]) / (vx[2] - vx[3]);
            }

            if ((double)x < vx[1]) {
                /* интерполяция */
                if (fabs(vx[0] - vx[1]) <= 0.01) {
                    if (vy[0]<vy[1]) ye = vy[1];
                    else if (vy[0]>vy[1]) ye = vy[0];
                    else     ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
                }
                else
                    ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
                
            }
            else {
                /* интерполяция */
                if (fabs(vx[1] - vx[2]) <= 0.01) {
                    if (vy[1]<vy[2]) ye = vy[2];
                    else if (vy[1]>vy[2]) ye = vy[1];
                    else     ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
                }
                else
                    ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
            }
            
            y = (int)ceil(ys);
        }
        
        // условие выхода
        if (x > vx[2]) break;
        
        ptsx[noPoints] = x;
        ptsy[noPoints] = y;
        noPoints++;
    }
    
    *pNoPoints = noPoints;
}

