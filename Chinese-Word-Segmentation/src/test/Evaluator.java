package test;

import java.util.ArrayList;


public class Evaluator {
	public int stdCount, rstCount, corCount;
	public long startTime;
	public Evaluator() {
		this.stdCount = 0;
		this.rstCount = 0;
		this.corCount = 0;
		this.startTime = System.currentTimeMillis();
	}
	
	public ArrayList<String> getWordSet(ArrayList<String> words) {
		int offSet = 0;
		ArrayList<String> wordSet = new ArrayList<String>();
		for (String w: words) {
			wordSet.add(offSet+w);
			offSet += w.length();
		}
		return wordSet;
	}
	
	public void call(ArrayList<String> std, ArrayList<String> rst) {
		ArrayList<String> stdWords = this.getWordSet(std);
		ArrayList<String> rstWords = this.getWordSet(rst);
		this.stdCount += stdWords.size();
		this.rstCount += rstWords.size();
		for (int i = 0; i < rstWords.size(); i++) {
			if (stdWords.contains(rstWords.get(i))) {
				this.corCount += 1;
			}
		}
	}
	
	public float report() {
		float time = (float)(System.currentTimeMillis()-this.startTime) / 1000;
		double precision = this.corCount / (double)this.rstCount;
		double recall = this.corCount / (double)this.stdCount;
		double fScore = 2*precision*recall / (precision+recall);
		System.out.printf("历时：%.2fs, 标准词数：%d, 分析词数：%d, 正确词数：%d\n",
				time, this.stdCount, this.rstCount, this.corCount);
		System.out.printf("P: %.6f\tR: %.6f\tF: %.6f\n", precision, recall, fScore);
		return time;
	}
}
