package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class TransForm {
	public static void main(String[] args) {
		String testPath = "data/vocab.large.txt";
		String resultPath = "data/train1+dict.txt";
		String line = null;
		BufferedReader br;
		try{
	        BufferedWriter writer = new BufferedWriter(
	        		new OutputStreamWriter(new FileOutputStream(resultPath), "utf-8"));  
			br = new BufferedReader( new InputStreamReader(new FileInputStream(testPath), "utf-8"));	
			while((line = br.readLine()) != null){
				String res = "";
				String[] words = line.split("\t");
				for (String s : words) {
					res += s + "";
				}
				res = words[0];
	            writer.write(res.trim()+'\n');
			}
            writer.flush();
            writer.close();
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}
}
