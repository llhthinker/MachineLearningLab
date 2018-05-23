package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import seg.CWS;
import seg.MMSeg;
import seg.APSeg;
import seg.TrainModel;

public class Main {
	
	public static void main(String[] args) {
//		String testFile = "data/test/final-test.2.txt";
//		String modelFile = "./data/model/model.txt";
//		String resultFile = "./data/res/ap-final-res.2.txt";
//		String plusTrainFile1 = "./data/gold/real_ans2.txt";
		String plusTrainFile2 = "./data/train/train_pku.txt";
//		String plusTrainFile3 = "./data/res/mm-final-res.2.txt";
//		String rightFile = "./data/res/mm-final-res.2.txt";
	
		String testFile = "./data/test/pku_test1.txt";
		String modelFile =  "./data/model/model1+pku.txt";
		String resultFile = "./data/res/pku_test1_res.txt";
		String mmResultFile = "./data/res/mm_pku_test_res1.txt";

		String rightFile = "./data/gold/pku_test1_gold.txt";

		String line;
		BufferedReader br;
		try{
			CWS cws = new CWS();
			cws.weights.loadModel(modelFile);
			long tmpTime = System.currentTimeMillis();
//			plusTrain (plusTrainFile2, cws, 1.0, 6);
//			plusTrain (plusTrainFile3, cws, 0.1, 14);	
			plusTrain (mmResultFile, cws, 0.08, 20);	
//			System.out.printf("处理完成，用时: %.2fs\n", 
//				(System.currentTimeMillis()-tmpTime) / 1000.0);
	        BufferedWriter writer = new BufferedWriter(
	        		new OutputStreamWriter(new FileOutputStream(resultFile), "utf-8"));  
			br = new BufferedReader( new InputStreamReader(new FileInputStream(testFile), "utf-8"));	
			MMSeg.Init();
			while((line = br.readLine()) != null){
					String res = "";
					ArrayList<String> mmRes = MMSeg.segment(line);
					ArrayList<String> resList = chooseFinalRes(mmRes, APSeg.segment(cws, line));
					
					for (String s : resList) {
						res += s + "\t";
					}
	            writer.write(res.trim()+'\n');
	        }
			
			System.out.println("分词完成:\n" + testFile + " -> " + resultFile);
			System.out.println("开始评测...");
			Judge.judge(resultFile, rightFile);
			
            writer.flush();
            writer.close();
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static ArrayList<String> chooseFinalRes(ArrayList<String> mmRes, ArrayList<String> apRes) {
		//如果分词结果词数不同，则取分词数量较少的那个
		if(mmRes.size() != apRes.size()){
			if(mmRes.size() > apRes.size())
				return apRes;
			else  {
			//	System.out.println(mmRes);
				return mmRes;
			}
		}
		//如果分词结果词数相同
		else{
			//如果分词结果相同，就说明没有歧义，可返回任意一个
			int i;
			int mmTotalFreq = 0, apTotalFreq = 0;
			boolean isSame = true;
			for(i = 0; i < mmRes.size();  i++){
				if (MMSeg.segDict.containsKey(mmRes.get(i)))
					mmTotalFreq += MMSeg.segDict.get(mmRes.get(i));
				if (MMSeg.segDict.containsKey(apRes.get(i)))
					apTotalFreq += MMSeg.segDict.get(apRes.get(i));
				if(!mmRes.get(i).equals(apRes.get(i)))
					isSame = false;
			}
			if(isSame)
				return mmRes;
			else{
				//分词结果不同，返回字频总数较大的
				if(mmTotalFreq > apTotalFreq) {
				//	System.out.println(mmRes);
					return mmRes;
				}
				else return apRes;
			}
		}
	}
	
	public static void plusTrain (String plusTrainFile, CWS cws, double delta, int iterNum) {
		String trainFile = plusTrainFile;
		float totalTime = 0;
		for (int i = 0; i < iterNum; i++) {
			System.out.println("第"+ (i+1) + "次迭代...");
			Evaluator evaluator = new Evaluator();
			BufferedReader br;
			String line;				
			try {
				br = new BufferedReader( new InputStreamReader(
						new FileInputStream(trainFile), "utf-8"));
				while((line = br.readLine()) != null){
					ArrayList<Integer> y = new ArrayList<Integer>();
					if (line.isEmpty())
						continue;
					String x = TrainModel.loadExample(line.split("\t"), y);
					ArrayList<Integer> z = cws.decode(x);
					evaluator.call(TrainModel.dumpExample(x, y), TrainModel.dumpExample(x, z));
					cws.weights.step += 1;
					if (!z.equals(y)) {
						cws.updateWeights(x, y, delta);  //正向激励
						cws.updateWeights(x, z, -1*delta);  //反向惩罚
					}
				}
				totalTime += evaluator.report();
				cws.weights.updateAll();
				cws.weights.average();
				cws.weights.unaverage();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		System.out.printf("训练总耗时: %.2fs\n", totalTime);
	}
}