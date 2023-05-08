#include "parallelFilter.h"


void parallelFilter::operator()(const Range& range) const
{
    for (int r = range.start; r < range.end; r++)
    {
        int i = r / m_src.cols, j = r % m_src.cols;
        double value = 0;

        if (j <= m_src.cols - m_kernel.cols && i <= m_src.rows - m_kernel.rows) {
            for (int k = -sz_rows; k <= sz_rows; k++)
            {
                const double* sptr = m_src.ptr<double>(i + sz_rows + k);
                for (int l = -sz_cols; l <= sz_cols; l++)
                {
                    /*cout << k + sz_rows << "," << l + sz_cols << ":" << m_kernel.ptr<double>(k + sz_rows)[l + sz_cols];
                    cout << "xuong dong\n";*/
                    value += m_kernel.ptr<double>(k + sz_rows)[l + sz_cols] * sptr[j + sz_cols + l];
                }
            }
            m_dst.ptr<double>(i)[j] = saturate_cast<double>(value);
        }
    }
}