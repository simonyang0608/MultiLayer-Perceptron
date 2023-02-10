using System;

namespace Perceptron_MLP
{
    public class Data_pipe
    {
        ////////random data sample parameters initialization////////
        private double[] sample_pts_norm;
        private double[] gtrth_pts_norm;

        private Random rand_sample_pts_norm;
        private double samp_sample_pts_norm;
        private double scl_sample_pts_norm;

        private Random rand_gtrth_pts_norm;
        private double samp_gtrth_pts_norm;
        private double scl_gtrth_pts_norm;

        public Data_pipe(int sample_pts_num)
        {
            sample_pts_norm = new double[sample_pts_num];
            gtrth_pts_norm = new double[sample_pts_num];

            rand_sample_pts_norm = new Random();
            rand_gtrth_pts_norm = new Random();
        }

        public double[] Assn_rand_samp(int sample_pts_num) //random assign input sample
        {
            for (int col_idx = 0; col_idx < sample_pts_num; col_idx++)
            {
                samp_sample_pts_norm = rand_sample_pts_norm.NextDouble(); //normalize to 0~1
                //scl_sample_pts = (samp_sample_pts * (sample_pts_num - 1) + 1);

                sample_pts_norm[col_idx] = samp_sample_pts_norm;
            }

            return sample_pts_norm;
        }

        public double[] Assn_rand_gtrth(int sample_pts_num) //random assign ground truth sample
        {
            for (int col_idx = 0; col_idx < sample_pts_num; col_idx++)
            {
                samp_gtrth_pts_norm = rand_gtrth_pts_norm.NextDouble(); //normalize to 0~1
                //scl_gtrth_pts = (samp_gtrth_pts * (sample_pts_num - 1) + 1);

                gtrth_pts_norm[col_idx] = samp_gtrth_pts_norm;
            }

            return gtrth_pts_norm;
        }
    }
}