package ece.cpen502.LearningAgents;

import ece.cpen502.Interface.CommonInterface;
import ece.cpen502.LUT.LearningAgent;
import ece.cpen502.LUT.RobotAction;
import ece.cpen502.NN.RLNeuralNet;

import java.io.File;
import java.io.IOException;

public class LearningAgentNN implements CommonInterface {
    public enum Algo{QLearn, Sarsa};

    private static double learningRate = 0.2;
    static int noOfHiddenNeurons = 15;
    static double  momentum = 0.3;
    private double discountFactor = 0.9;
    private double[]prevState = new double[6];
    private int prevAction = -1;
    private double[][] inputToHiddenWeights, hiddenToOutputWeights; //+1 to accommodate a bias weight
    public static RLNeuralNet nn = new RLNeuralNet(learningRate, momentum, noOfHiddenNeurons, false, null, "LUTNN_Weights.txt");


//    RLNeuralNet nn = new RLNeuralNet(learningRate, momentum, noOfHiddenNeurons, false, null, "LUTNN_Weights.txt", inputToHiddenWeights, hiddenToOutputWeights);
    @Override
    public double train(double [] X, double argValue) { return 0; };

    public void train(double[] curState, int curAction, double reward, Algo algo) {
        if (prevState != null && prevAction != -1) {
            // prev state, prev action -> Q value
            double Q = nn.getQ(prevState,prevAction);
            switch (algo) {
                case QLearn:
                    Q += this.learningRate * (reward + this.discountFactor * nn.getMaxQ(curState) - Q);
                    break;
                case Sarsa:
                    Q += this.learningRate * (reward + this.discountFactor * nn.getQ(curState, curAction) - Q);
                    break;
            }
            nn.backPropagation(prevState,prevAction,Q);
        }
        prevState = curState;
        prevAction = curAction;
    }

    public int getAction(double[] state, double epsilon){
        if (Math.random() > epsilon)
            return nn.getOptimalAction(state);
        return (int) (Math.random() * RobotAction.actionsCount);
    }

    @Override
    public void save(File argFile) {}

    @Override
    public void load(String argFileName) throws IOException {

    }

    public double outputFor(double[] X) { return 0; }
}
