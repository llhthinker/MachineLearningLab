package test;

import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import seg.APSeg;
import seg.CWS;
import seg.MMSeg;

public class Demo {
	public static void main(String[] args) {

		String modelFile = "./data/model/model1_pku.txt";
		modelFile = "./data/model/model1+pku.txt";
		CWS cws = new CWS();
		cws.weights.loadModel(modelFile);
		MMSeg.Init();
		String testLine;
		//  为人民办实事
		//  研究生命起源
		//  结婚的和尚未穿袈裟
		//  他从马上下来
		//  王毅回应蔡英文与特朗普通话
		//	他叫汤姆去拿外衣。
		//	他点头表示同意我的意见。
		//	我们即将以昂扬的斗志迎来新的一年。
		//	国内专家学者40余人参加研讨会。
		System.out.println("请输入测试语句：");
		Scanner in=new Scanner(System.in);
		while(!(testLine = in.nextLine()).equals("end")) {
			ArrayList<String> mmRes= MMSeg.segment(testLine);
			
			ArrayList<String> apRes = APSeg.segment(cws, testLine);
			System.out.println("最大双向匹配分词结果:");
			System.out.println(mmRes);
			System.out.println("平均感知器分词结果");
			System.out.println(apRes);
			System.out.println("请输入测试语句：");
		}
	}
}
