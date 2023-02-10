using System;

namespace Perceptron_MLP
{
    public class Perceptron
    {
        ////////random MLP network parameters initialization////////
        private Random rand_wgts_inpt_hiddn;
        private double samp_wgts_inpt_hiddn;
        private double[,] weights_inpt_hiddn;

        private Random rand_wgts_hiddn_opt;
        private double samp_wgts_hiddn_opt;
        private double[,] weights_hiddn_opt;

        private Random rand_biass_hiddn;
        private double samp_biass_hiddn;
        private double[] biases_hiddn;

        private Random rand_biass_opt;
        private double samp_biass_opt;
        private double[] biases_opt;

        private double tmp_weights_hiddn_val;
        private double tmp_biases_hiddn_val;

        private double[] hidden_node;
        private double out_pred = 0.0;

        public Perceptron(int inpt_nde_num, int hiddn_nde_num)
        {
            weights_inpt_hiddn = new double[inpt_nde_num, hiddn_nde_num];
            weights_hiddn_opt = new double[hiddn_nde_num, inpt_nde_num];

            biases_hiddn = new double[hiddn_nde_num];
            biases_opt = new double[inpt_nde_num];

            rand_wgts_inpt_hiddn = new Random();
            rand_wgts_hiddn_opt = new Random();

            rand_biass_hiddn = new Random();
            rand_biass_opt = new Random();

            hidden_node = new double[hiddn_nde_num];

            for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++) //assign 0-values for hidden node
            {
                hidden_node[col_idx] = 0.0;
            }
        }

        public void Assn_rand_wgts(int inpt_nde_num, int hiddn_nde_num)
        {
            /////////weights between input node and hidden node/////////
            for (int row_idx = 0; row_idx < inpt_nde_num; row_idx++)
            {
                for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
                {
                    samp_wgts_inpt_hiddn = (-1 + (2 * rand_wgts_inpt_hiddn.NextDouble())); //normalize -1~1

                    weights_inpt_hiddn[row_idx, col_idx] = samp_wgts_inpt_hiddn;
                }
            }

            /////////weights between hidden node and output node/////////
            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++)
            {
                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    samp_wgts_hiddn_opt = (-1 + (2 * rand_wgts_hiddn_opt.NextDouble())); //normalize -1~1

                    weights_hiddn_opt[row_idx, col_idx] = samp_wgts_hiddn_opt;
                }
            }
        }

        public void Assn_rand_bias(int inpt_nde_num, int hiddn_nde_num)
        {
            /////////bias in hidden node/////////
            for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
            {
                samp_biass_hiddn = (-1 + (2 * rand_biass_hiddn.NextDouble())); //normalize -1~1

                biases_hiddn[col_idx] = samp_biass_hiddn;
            }

            /////////bias in output node/////////
            for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
            {
                samp_biass_opt = (-1 + (2 * rand_biass_opt.NextDouble())); //normalize -1~1

                biases_opt[col_idx] = samp_biass_opt;
            }
        }

        public double Feedforward_pass(int inpt_nde_num, int hiddn_nde_num, int data_pts_idx, double[] sample_pts)
        {
            ///////stage 1//////
            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //input node and hidden node
            {
                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    hidden_node[row_idx] += (sample_pts[data_pts_idx] * weights_inpt_hiddn[col_idx, row_idx]);
                }

                hidden_node[row_idx] += biases_hiddn[row_idx];
            }


            ///////stage 2//////
            for (int row_idx = 0; row_idx < inpt_nde_num; row_idx++) //hidden node and output node
            {
                for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
                {
                    out_pred += (hidden_node[col_idx] * weights_hiddn_opt[col_idx, row_idx]);
                }

                out_pred += biases_opt[row_idx];
            }

            return out_pred;
        }

        public void Backpropagation_pass(int inpt_nde_num, int hiddn_nde_num, int data_pts_idx, double[] sample_pts, double[] gtrth_pts, double learning_rate)
        {
            ///////stage 1//////
            for (int row_idx = 0; row_idx < inpt_nde_num; row_idx++)  //weights variation between input node and hidden node
            {
                for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
                {
                    tmp_weights_hiddn_val = 0.0;

                    for (int opt_nde_idx = 0; opt_nde_idx < inpt_nde_num; opt_nde_idx++)
                    {
                        tmp_weights_hiddn_val += (learning_rate * (2 / inpt_nde_num) * (gtrth_pts[data_pts_idx] - out_pred) * (-1) * weights_hiddn_opt[col_idx, opt_nde_idx]);
                    }

                    weights_inpt_hiddn[row_idx, col_idx] -= (tmp_weights_hiddn_val * sample_pts[data_pts_idx]); //update weight parameters
                }
            }

            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //biases variation in hidden node
            {
                tmp_biases_hiddn_val = 0.0;

                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    tmp_biases_hiddn_val += (learning_rate * (2 / inpt_nde_num) * (gtrth_pts[data_pts_idx] - out_pred) * (-1) * weights_hiddn_opt[row_idx, col_idx]);
                }

                biases_hiddn[row_idx] -= tmp_biases_hiddn_val; //update bias parameters
            }


            ///////stage 2//////
            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //weights variation between hidden node and output node
            {
                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    weights_hiddn_opt[row_idx, col_idx] -= (learning_rate * ((2 / inpt_nde_num) * (gtrth_pts[data_pts_idx] - out_pred) * (-1) * hidden_node[row_idx])); //update weight parameters
                }
            }

            for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++) //biases variation in output node
            {
                biases_opt[col_idx] -= (learning_rate * (2 / inpt_nde_num) * (gtrth_pts[data_pts_idx] - out_pred) * (-1)); //update bias parameters
            }

            out_pred = 0.0; //clear output predict

            Array.Clear(hidden_node, 0, hidden_node.Length); //clear hidden node to 0-values
        }

        public double[,] Updated_wgts_inpt_hiddn()
        {
            return weights_inpt_hiddn;
        }

        public double[,] Updated_wgts_hiddn_opt()
        {
            return weights_hiddn_opt;
        }

        public double[] Updated_bias_hiddn()
        {
            return biases_hiddn;
        }

        public double[] Updated_bias_opt()
        {
            return biases_opt;
        }
    }
}