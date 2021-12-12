package ece.cpen502.LUT;

import ece.cpen502.Interface.LUTInterface;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

public class LookupTable implements LUTInterface {
    private int numStates = RobotState.statesCount;
    private int numActions = RobotAction.actionsCount;
    public double[][] lookupTable;
    public  int robotID;

    public LookupTable(){
        this.lookupTable = new double[this.numStates][this.numActions];
        this.initialiseLUT();
        robotID = new Random().nextInt();
    }

    public void set(int state, int action, double value){
        this.lookupTable[state][action] = value;
    }

    public double get(int state, int action){
        return this.lookupTable[state][action];
    }

    public double getMax(int state){
        double max = Double.NEGATIVE_INFINITY;
        for (double QValue: this.lookupTable[state])
            max = Math.max(QValue, max);
        return max;
    }

    public int getOptimalAction(int state){
        int idx = 0;
        for (double QValue: this.lookupTable[state]){
            if (QValue == this.getMax(state)) return idx;
            ++idx;
        }
        return 0;
    }

    public void save(File argFile) {
        SimpleDateFormat sdf = new SimpleDateFormat("File-ddMMyy-hhmmss.SSS.txt");
        String fileName = sdf.format(new Date());

        try{
            ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(fileName));
            outputStream.writeObject(getLutTable());
        }catch(Exception e){
            System.out.println(e);
        }
    }

    public void load(String argFileName) throws IOException {}

    @Override
    public void initialiseLUT() {
        for (int i = 0; i < this.numStates; ++i)
            for (int j = 0; j < this.numActions; ++j)
                this.lookupTable[i][j] = 0.0;
    }

    public double[][] getLutTable(){
        return lookupTable;
    }
}
