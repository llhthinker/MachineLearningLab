package seg;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import test.Judge;


public class MMSeg {
	public static Map<String, Integer> segDict;
	private final static int MAX_WORD_LEN = 20;
	
	//繁体问题
	public static void main(String [] args){
		MMSeg.Init();
		String testPath = "data/test/test1.txt";
		String resultPath = "data/res/result1.txt";
		String rightPath = "data/gold/real_ans1.txt";
		
//		String testPath = "data/test/test2.txt";
//		String resultPath = "data/res/mm_result2.txt";
//		String rightPath = "data/gold/real_ans2.txt";
		String line = null;
		BufferedReader br;
		try{
	        BufferedWriter writer = new BufferedWriter(
	        		new OutputStreamWriter(new FileOutputStream(resultPath), "utf-8"));  
			br = new BufferedReader( new InputStreamReader(new FileInputStream(testPath), "utf-8"));	
			while((line = br.readLine()) != null){
				String res = "";
				for (String s : MMSeg.segment(line)) {
					res += s + "\t";
				}
	            writer.write(res.trim()+'\n');
			}
			System.out.println("分词完成:\n" + testPath + " -> " + resultPath);

			Judge.judge(resultPath, rightPath);
			
            writer.flush();
            writer.close();
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}

	public static void readDictFile(String dicPath, int maxWordLen, String kind) {
		BufferedReader br;
		String line;
		try{
			br = new BufferedReader( new InputStreamReader(new FileInputStream(dicPath), "utf-8"));	
			while((line = br.readLine()) != null){
				line = line.trim();
				String[] words = line.split("\t");
				String word = words[0];
				int freq = 1;
				if (words.length == 2)
					freq = Integer.parseInt(line.split("\t")[1]);
				if(word.isEmpty())
					continue;
				if (!segDict.containsKey(word) && word.length() <= maxWordLen) {
					
					if (kind.equals("name")) {
						segDict.put(word.substring(0, 1), freq);
						segDict.put(word.substring(1), freq);
					}
					else {
						segDict.put(word, freq);
					}
				}
			}
		}catch(IOException e){
			e.printStackTrace();
		}

	}
	//加载词典
	public static void Init(){
		segDict = new HashMap<String, Integer>();
		String dicPath = "data/vocab.large.txt";
		String dicPathWebsite = "data/vocab.large.plus.website.txt";
		String dicPathNet = "data/vocab.large.plus.net.txt";
		String dicPathName = "data/vocab.large.plus.name.txt";
		String dicPathClassic = "data/vocab.large.plus.classic.txt";
		String dicPathGame = "data/vocab.large.plus.game.txt";

		readDictFile(dicPath, 100, "common");
//		readDictFile(dicPathWebsite, 4, "website");
//		readDictFile(dicPathNet, 2, "net");
//		readDictFile(dicPathName, 3, "name");
//		readDictFile(dicPathClassic, 4, "classic");
//		readDictFile(dicPathGame, 4, "game");
//		
	}
	/**
	 * 前向算法分词
	 * @param segDict 分词词典
	 * @param phrase 待分词句子
	 * @return 前向分词结果
	 */
	private static Vector<String> FMM( String  phrase){
		int maxlen = MAX_WORD_LEN;
		Vector<String> fmmList = new Vector<String>();
		int len_phrase = phrase.length();
		int i=0, j=0;
		
		while(i < len_phrase){
			int end = i+maxlen;
			if(end >= len_phrase)
				end = len_phrase;
			String phrase_sub = phrase.substring(i, end);
			for(j = phrase_sub.length(); j >=0; j--){
				if(j == 1)
					break;
				String key =  phrase_sub.substring(0, j);
				if(segDict.containsKey(key.toUpperCase())){
					fmmList.add(key);
					i += key.length() -1;
					break;
				}
			}
			if(j == 1)
				fmmList.add(""+phrase_sub.charAt(0));
			i += 1;
		}
		return fmmList;
	}
	
	/**
	 * 后向算法分词
	 * @param segDict 分词词典
	 * @param phrase 待分词句子
	 * @return 后向分词结果
	 */
	private static Vector<String> BMM( String  phrase){
		int maxlen = MAX_WORD_LEN;
		Vector<String> bmmList = new Vector<String>();
		int len_phrase = phrase.length();
		int i=len_phrase,j=0;
		
		while(i > 0){
			int start = i - maxlen;
			if(start < 0)
				start = 0;
			String phrase_sub = phrase.substring(start, i);
			for(j = 0; j < phrase_sub.length(); j++){
				if(j == phrase_sub.length()-1)
					break;
				String key =  phrase_sub.substring(j);
				if(segDict.containsKey(key)){
					bmmList.insertElementAt(key, 0);
					i -= key.length() -1;
					break;
				}
			}
			if(j == phrase_sub.length() -1)
				bmmList.insertElementAt(""+phrase_sub.charAt(j), 0);
			i -= 1;
		}
		return bmmList;
	}
		
	/**
	 * 该方法结合正向匹配和逆向匹配的结果，得到分词的最终结果
	 * @param FMM 正向匹配的分词结果
	 * @param BMM 逆向匹配的分词结果
	 * @param return 分词的最终结果
	 */
	public static ArrayList<String> subSeg(String phrase){
		ArrayList<String> fmmList = new ArrayList<String>();
		ArrayList<String> bmmList = new ArrayList<String>();
		fmmList.addAll(FMM(phrase));
		bmmList.addAll(BMM(phrase));
		//如果正反向分词结果词数不同，则取分词数量较少的那个
		if(fmmList.size() != bmmList.size()){
			if(fmmList.size() > bmmList.size())
				return bmmList;
			else return fmmList;
		}
		//如果分词结果词数相同
		else{
			//如果正反向的分词结果相同，就说明没有歧义，可返回任意一个
			int i;
			int fTotalFreq = 0, bTotalFreq = 0;

			boolean isSame = true;
			for(i = 0; i < fmmList.size();  i++){
				if (segDict.containsKey(fmmList.get(i)))
					fTotalFreq += segDict.get(fmmList.get(i));
				if (segDict.containsKey(bmmList.get(i)))
					bTotalFreq += segDict.get(bmmList.get(i));
				if(!fmmList.get(i).equals(bmmList.get(i)))
					isSame = false;
			}
			if(isSame)
				return fmmList;
			else{
				//分词结果不同，返回字频总数较大的
				if(fTotalFreq > bTotalFreq)
					return fmmList;
				else return bmmList;
			}
		}
	}
	
	public static String[] preProcess(String phrase, ArrayList<String> urls) {
		String regex = "(http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?)" // urls
				+ "|([0-9][0-9]):([0-9][0-9])(:[0-9][0-9])?"    //时间
				+ "|([0-9]+)/([0-9][0-9])/([0-9][0-9])"    //日期
				+ "|([0-9\\.]+)"                                         // 数字
				+ "|([a-z|A-Z|0-9|_|-]+)"                                         // 非汉字
				+ "|//"
				//+ "|\\[|【|@|]|#|】|\\]|\""
				+ "|[-|。|，]+"
			//	+ "０|１|２|３|４|５|６|７|８|９]+"
				;                  
		Pattern pattern = Pattern.compile(regex);
		Matcher matcher = pattern.matcher(phrase);
		int firstCount = 0;
		while (matcher.find()) {
			//System.out.println(matcher.group());
			urls.add(matcher.group());
		}
		String[] ss = phrase.split(regex);
		return ss;
	}
	
	public static ArrayList<String> segment(String phrase) {
		ArrayList<String> mmList = new ArrayList<String>();
		ArrayList<String> urls = new ArrayList<String>();
		String[] subStrs = preProcess(phrase, urls);
		for (int i = 0; i < subStrs.length; i++) {
			mmList.addAll(subSeg(subStrs[i]));
			if (i < urls.size()) {
				mmList.add(urls.get(i));
			}
		}
		int j = subStrs.length;
		if (j < urls.size()) {
			mmList.add(urls.get(j));
			j++;
		}
		return mmList;
	}

} 