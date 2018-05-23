package seg;

import java.util.ArrayList;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class APSeg {
	public static ArrayList<String> subSeg(CWS cws, String line){
		ArrayList<Integer> y = new ArrayList<Integer>();
		String x = TrainModel.loadExample(line.split(""), y);
        ArrayList<Integer> z = cws.decode(x);
		ArrayList<String> wordList = TrainModel.dumpExample(x, z);
		return wordList;
	}
	
	public static String[] preProcess(String phrase, ArrayList<String> urls) {
		String regex = "(http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?)"// urls
//				+ "|([0-9\\.]+)"                                         // 数字
				+ "|([a-z|A-Z|0-9|_]+)"                                         // 非汉字
//				+ "|//"
//				+ "|[【|@]"
//				+ "|[0-9]+:[0-9]+(:[0-9]+)?"                      //时间
				;
		Pattern pattern = Pattern.compile(regex);
		Matcher matcher = pattern.matcher(phrase);
		int firstCount = 0;
		while (matcher.find()) {
			//System.out.println(matcher.group());
			urls.add(matcher.group());
			if (matcher.start() == 0) {
				firstCount += 1;
			}
		}
		urls.add("firstCount"+firstCount);
		String[] ss = phrase.split(regex);
		return ss;
	}
	
	public static ArrayList<String> segment(CWS cws, String phrase) {
		ArrayList<String> resList = new ArrayList<String>();
		ArrayList<String> urls = new ArrayList<String>();
		String[] subStrs = preProcess(phrase, urls);

		String firstCountS = urls.remove(urls.size()-1);
		int firstCount = Integer.valueOf(firstCountS.charAt(firstCountS.length()-1)) 
				- Integer.valueOf('0');
		int firstI = firstCount;
		if (firstI > 0) {
			resList.add(urls.get(0));
			//firstI--;
		}
		for (int i = 0; i < subStrs.length; i++) {
			if (subStrs[i].isEmpty()) 
				continue;
			resList.addAll(subSeg(cws, subStrs[i]));
			if (i >= firstCount && i < urls.size()) {
				resList.add(urls.get(i));
			}
		}
		int j = subStrs.length;
		if (j < urls.size()) {
			resList.add(urls.get(j));
			j++;
		}
		return resList;
	}
}
