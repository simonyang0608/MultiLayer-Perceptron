using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Perceptron_MLP
{
    public partial class Form1 : Form
    {
        Data_pipe Data_flow;

        Perceptron MLP;

        ////////hyper parameters////////
        private double[,] init_weights_inpt_hiddn;
        private double[,] init_weights_hiddn_opt;
        private double[] init_biases_hiddn;
        private double[] init_biases_opt;

        private double[,] upd_weights_inpt_hiddn;
        private double[,] upd_weights_hiddn_opt;
        private double[] upd_biases_hiddn;
        private double[] upd_biases_opt;

        ////////optimizer parameters////////
        private double[] loss_rcd;
        private double learning_rate;
        private double loss = 8.79;
        private double out_pred;

        private int iterations;

        ////////perceptron node parameters////////
        private int inpt_nde_num = 1;
        private int hiddn_nde_num;

        ////////data sample parameters////////
        private double[] sample_pts_norm;
        private double[] gtrth_pts_norm;
        private double[] outpred_pts;

        private int sample_pts_num;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void timer1_Tick(object sende, EventArgs e)
        {
            /////////display object initialization/////////
            textBox4.Clear();
            textBox5.Clear();

           
            //////////////////////////////
            ////////start learning////////
            //////////////////////////////
            for (int data_pts_idx = 0; data_pts_idx < sample_pts_num; data_pts_idx++) //loop for whole sample points
            {
                out_pred = MLP.Feedforward_pass(inpt_nde_num, hiddn_nde_num, data_pts_idx, sample_pts_norm); //feed-forward pass

                MLP.Backpropagation_pass(inpt_nde_num, hiddn_nde_num, data_pts_idx, sample_pts_norm, gtrth_pts_norm, learning_rate); //back-propagation pass

                outpred_pts[data_pts_idx] = out_pred; //loss record

                loss_rcd[data_pts_idx] = (Math.Pow((gtrth_pts_norm[data_pts_idx] - out_pred), 2)); //loss calculation for MSE loss function
            }

            iterations++;

            loss = loss_rcd.Average(); //calculate average loss per whole sample

            upd_weights_inpt_hiddn = MLP.Updated_wgts_inpt_hiddn(); //input-hidden node weights update
            upd_weights_hiddn_opt = MLP.Updated_wgts_hiddn_opt(); //hidden-output node weights update
            upd_biases_hiddn = MLP.Updated_bias_hiddn(); //hidden node bias update
            upd_biases_opt = MLP.Updated_bias_opt(); //output node bias update


            /////////iterations and loss visualization/////////
            textBox4.AppendText(iterations.ToString());
            textBox5.AppendText(Math.Round(loss, 7).ToString());


            /////////dynamic chart update (line chart for loss v.s. iterations)/////////
            chart1.ChartAreas[0].AxisY.Maximum = (int)(loss_rcd.Max() + 1);
            chart1.ChartAreas[0].AxisY.Minimum = (int)(loss_rcd.Min() - 1);

            chart1.ChartAreas[0].RecalculateAxesScale();

            chart1.Series[0].Points.AddXY(iterations, loss);


            /////////dynamic chart update (line & point chart for truth v.s. learned)/////////
            chart2.Series[0].Points.Clear();
            chart2.Series[1].Points.Clear();

            chart2.ChartAreas[0].AxisY.Maximum = (int)Math.Max((gtrth_pts_norm.Max()*sample_pts_num + 1), (outpred_pts.Max()*sample_pts_num + 1));
            chart2.ChartAreas[0].AxisY.Minimum = (int)Math.Min((gtrth_pts_norm.Min()*sample_pts_num - 1), (outpred_pts.Min()*sample_pts_num - 1));

            chart2.ChartAreas[0].RecalculateAxesScale();

            for (int data_pts_idx = 0; data_pts_idx < sample_pts_num; data_pts_idx++) //plot ground truth data points
            {
                chart2.Series[0].Points.AddXY(data_pts_idx, (gtrth_pts_norm[data_pts_idx]*sample_pts_num));
            }

            for (int data_pts_idx = 0; data_pts_idx < sample_pts_num; data_pts_idx++) //plot output predic data line
            {
                chart2.Series[1].Points.AddXY(data_pts_idx, (outpred_pts[data_pts_idx]*sample_pts_num));
            }


            /////////dynamic chart update (point chart for weights v.s. biases)/////////
            chart3.Series[0].Points.Clear();
            chart3.Series[1].Points.Clear();

            chart3.ChartAreas[0].RecalculateAxesScale();

            for (int row_idx = 0; row_idx < inpt_nde_num; row_idx++) //weights parameters (input-hidden)
            {
                for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
                {
                    chart3.Series[0].Points.AddY(upd_weights_inpt_hiddn[row_idx, col_idx]);
                }
            }

            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //bias parameters (hidden)
            {
                chart3.Series[1].Points.AddY(upd_biases_hiddn[row_idx]);
            }

            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //weight parameters (hidden-output)
            {
                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    chart3.Series[0].Points.AddY(upd_weights_hiddn_opt[row_idx, col_idx]);
                }
            }

            for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++) //bias parameters (output)
            {
                chart3.Series[1].Points.AddY(upd_biases_opt[col_idx]);
            }

            if (loss < 8.5E-8) { timer1.Enabled = false; button1.Enabled = true; }

            return;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            /////////////////////////////////////////////////////////////
            ////////throw error when textbox text is empty string////////
            /////////////////////////////////////////////////////////////
            if (textBox1.Text == string.Empty)
            {
                MessageBox.Show("Please input the number of input node", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

                return;
            }
            if (textBox2.Text == string.Empty)
            {
                MessageBox.Show("Please input the number of hidden node", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

                return;
            }
            if (textBox3.Text == string.Empty)
            {
                MessageBox.Show("Please input the learning rate value", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

                return;
            }


            //////////////////////////////////////////////
            ////////initialized related parameters////////
            //////////////////////////////////////////////
            /* values & data query from input textbox */
            sample_pts_num = int.Parse(textBox1.Text);
            hiddn_nde_num = int.Parse(textBox2.Text);
            learning_rate = double.Parse(textBox3.Text);

            /* random sample & ground truth data point saaignment */
            Data_flow = new Data_pipe(sample_pts_num);
            sample_pts_norm = Data_flow.Assn_rand_samp(sample_pts_num);
            gtrth_pts_norm = Data_flow.Assn_rand_gtrth(sample_pts_num);

            /* random weights & biases data point assignment */
            MLP = new Perceptron(inpt_nde_num, hiddn_nde_num);
            MLP.Assn_rand_wgts(inpt_nde_num, hiddn_nde_num);
            MLP.Assn_rand_bias(inpt_nde_num, hiddn_nde_num);

            init_weights_inpt_hiddn = MLP.Updated_wgts_inpt_hiddn();
            init_weights_hiddn_opt = MLP.Updated_wgts_hiddn_opt();
            init_biases_hiddn = MLP.Updated_bias_hiddn();
            init_biases_opt = MLP.Updated_bias_opt();

            /* loss & prediction record */
            outpred_pts = new double[sample_pts_num];
            loss_rcd = new double[sample_pts_num];


            //////////////////////////////////////////////////////////////////////////////////////////////
            ////////initially dynamic chart update (line & point chart for truth v.s. learned)////////////
            //////////////////////////////////////////////////////////////////////////////////////////////
            chart2.Series[0].Points.Clear();
            chart2.Series[1].Points.Clear();

            chart2.ChartAreas[0].AxisY.Maximum = (int)Math.Max((gtrth_pts_norm.Max()*sample_pts_num), (sample_pts_norm.Max()*sample_pts_num)) + 1;
            chart2.ChartAreas[0].AxisY.Minimum = (int)Math.Min((gtrth_pts_norm.Min()*sample_pts_num), (sample_pts_norm.Min()*sample_pts_num)) - 1;

            chart2.ChartAreas[0].RecalculateAxesScale();

            for (int data_pts_idx = 0; data_pts_idx < sample_pts_num; data_pts_idx++) //plot ground truth data points
            {
                chart2.Series[0].Points.AddXY(data_pts_idx, (gtrth_pts_norm[data_pts_idx]*sample_pts_num));
            }

            for (int data_pts_idx = 0; data_pts_idx < sample_pts_num; data_pts_idx++) //plot input sample data line
            {
                chart2.Series[1].Points.AddXY(data_pts_idx, (sample_pts_norm[data_pts_idx]*sample_pts_num));
            }


            //////////////////////////////////////////////////////////////////////////////////////
            /////////initially dynamic chart update (point chart for weights v.s. biases)/////////
            //////////////////////////////////////////////////////////////////////////////////////
            chart3.Series[0].Points.Clear();
            chart3.Series[1].Points.Clear();

            chart3.ChartAreas[0].RecalculateAxesScale();

            for (int row_idx = 0; row_idx < inpt_nde_num; row_idx++) //weights parameters (input-hidden)
            {
                for (int col_idx = 0; col_idx < hiddn_nde_num; col_idx++)
                {
                    chart3.Series[0].Points.AddY(init_weights_inpt_hiddn[row_idx, col_idx]);
                }
            }

            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //bias parameters (hidden)
            {
                chart3.Series[1].Points.AddY(init_biases_hiddn[row_idx]);
            }

            for (int row_idx = 0; row_idx < hiddn_nde_num; row_idx++) //weight parameters (hidden-output)
            {
                for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++)
                {
                    chart3.Series[0].Points.AddY(init_weights_hiddn_opt[row_idx, col_idx]);
                }
            }

            for (int col_idx = 0; col_idx < inpt_nde_num; col_idx++) //bias parameters (output)
            {
                chart3.Series[1].Points.AddY(init_biases_opt[col_idx]);
            }


            //////////////////////////////////////////////
            ////////initialized related objects///////////
            //////////////////////////////////////////////
            chart1.Series[0].Points.Clear();

            iterations = 0;

            timer1.Enabled = true;
            button1.Enabled = false;
        }
    }
}
