package seg;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class Weight {
	public Map<String, Double> values;
	public Map<String, Double> backup;
	public Map<String, Double> acc;
	public Map<String, Integer> lastStep;

	public int step;
	public double ld;  //惩罚因子
	public double p;  
	public double logP;

	public Weight() {
		this.values = new HashMap<String, Double>();
		this.backup = new HashMap<String, Double>();
		this.lastStep = new HashMap<String, Integer>();
		this.acc = new HashMap<String, Double>();
		this.step = 0;
		this.ld = 0.00001;  
		this.p = 0.9999;
		this.logP = Math.log(p);
	}
	
	public double noRegular(String key) {
		int dStep = this.step - this.lastStep.get(key);
		double value = this.values.get(key);
		double newValue = value;
		this.acc.put(key, this.acc.get(key)+dStep*value); 
		this.values.put(key, newValue);
		this.lastStep.put(key, this.step);
		return newValue;
	}
	
	public double L1Regular(String key) {
		if (!this.lastStep.containsKey(key)) {
			this.lastStep.put(key, 0);
		}
		int dStep = this.step - this.lastStep.get(key);
		double dvalue = dStep * this.ld;
		double value = this.values.get(key);
		double newValue = -1.0;
		if (value > 0) {
			newValue = 1.0;
		}
		newValue *= Math.max(0.0, Math.abs(value)-dvalue);
		if (newValue == 0.0) {
			if(!this.acc.containsKey(key)) {
				this.acc.put(key, this.values.get(key));
			}
			this.acc.put(key, this.acc.get(key)
					+ value * (value/this.ld) / 2);
		}
		else {
			if(!this.acc.containsKey(key)) {
				this.acc.put(key, this.values.get(key));
			}
			this.acc.put(key, this.acc.get(key) 
					+ (value + newValue) * dStep / 2);
		}
		this.values.put(key, newValue);
		this.lastStep.put(key, this.step);
		return newValue;
	}

	public double L2Regular(String key) {
		int dStep = this.step - this.lastStep.get(key);
		double value = this.values.get(key);
		double newValue = value * Math.exp(dStep * this.logP);
		this.acc.put(key, this.acc.get(key)
				+ value*(1-Math.exp(dStep*this.logP)/(1-this.p))); 
		this.values.put(key, newValue);
		this.lastStep.put(key, this.step);
		return newValue;
	}
	
	public void updateAll() {
		for (String key: this.values.keySet()) {
			this.L1Regular(key);
		}
	}
	
	public void updateWeights(String key, double delta) {
		if (!this.values.containsKey(key)) {
			this.values.put(key, 0.0);
			this.acc.put(key, 0.0);
			this.lastStep.put(key, this.step);
		}
		else {
			this.L1Regular(key);
		}
		double v = this.values.get(key);
		this.values.put(key, v+delta);
	}
	
	public void average() {
		this.backup.putAll(this.values);
		for (String key: this.acc.keySet()) {
			this.values.put(key, this.acc.get(key) / this.step);
		}
	}
	
	public void unaverage() {
		this.values.putAll(this.backup);
		this.backup.clear();
	}
	
	public void saveModel(String filename) {
		try{
	        BufferedWriter writer = new BufferedWriter(
	        		new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));  
			if (this.values.isEmpty())
				System.out.println("值为空");
			Set<Entry<String, Double>> set = this.values.entrySet();
		    Iterator<Entry<String, Double>> iter = set.iterator();
		    while(iter.hasNext()){
		    	Entry<String, Double> entry = iter.next(); 
		    	writer.write(entry.getKey()+":"+entry.getValue()+'\n');
		    }
            writer.flush();
            writer.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void loadModel(String filename) {
		BufferedReader br;
		String line;
		try {
			br = new BufferedReader( new InputStreamReader(
					new FileInputStream(filename), "utf-8"));
			while((line = br.readLine()) != null){
				String[] tmpValues = line.split(":");
				if (tmpValues.length == 2)
					this.values.put(tmpValues[0], Double.valueOf(tmpValues[1].trim()));
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//this.lastStep = null;
	}
	
	public double getValue(String key, double defaultValue) {
		if (!this.values.containsKey(key))
			return defaultValue;
		if (this.lastStep == null)
			return this.values.get(key);
		return this.L1Regular(key);
	}
}
