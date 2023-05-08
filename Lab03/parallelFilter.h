#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class parallelFilter : public ParallelLoopBody
{
private:
    Mat m_src, & m_dst;
    Mat m_kernel;
    int sz_rows, sz_cols;
public:
    parallelFilter(Mat src, Mat& dst, Mat kernel)
        : m_src(src), m_dst(dst), m_kernel(kernel)
    {
        sz_rows = kernel.rows / 2;
        sz_cols = kernel.cols / 2;
        m_dst = Mat(src.rows - kernel.rows + 1, src.cols - kernel.cols + 1, src.type());
    }
    virtual void operator()(const Range& range) const CV_OVERRIDE;
};