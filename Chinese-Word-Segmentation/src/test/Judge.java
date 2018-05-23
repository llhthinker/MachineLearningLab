package test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;


public class Judge {

	public static void main(String [] args){		
//		String testFile = "data/result2.txt";
//		String rightFile = "data/real_ans2.txt";
//		judge(testFile, rightFile);
		
	}
	
	public static float judge(String testFile, String rightFile) {
		ArrayList<String> testList = new ArrayList<String>();
		ArrayList<String> rightList = new ArrayList<String>();
		String line1, line2;
		BufferedReader br1, br2;
		try {
			br1 = new BufferedReader( new InputStreamReader(new FileInputStream(testFile), "utf-8"));
			br2 = new BufferedReader( new InputStreamReader(new FileInputStream(rightFile), "utf-8"));	
			while((line1 = br1.readLine()) != null){
				testList.add(line1);
			}
			while((line2 = br2.readLine()) != null){
				rightList.add(line2);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Evaluator evaluator = new Evaluator();
		for (int i = 0; i < testList.size(); i++) {
			String[] testWordList = testList.get(i).split("\t");
			String[] rightWordList = rightList.get(i).split("\t");
			ArrayList<String> rst = new ArrayList<String>();
			rst.addAll(Arrays.asList(testWordList));
			ArrayList<String> std = new ArrayList<String>();
			std.addAll(Arrays.asList(rightWordList));
			evaluator.call(std, rst);
		}
		return evaluator.report();
	}
}
