package seg;

import java.util.ArrayList;
import java.util.Arrays;

public class CWS {
	public Weight weights;
	
	public CWS() {
		this.weights = new Weight();
	}
	
	public ArrayList<String[]> enumFeatures(String x) {
		ArrayList<String[]> features = new ArrayList<String[]>();
		for (int i = 0; i < x.length(); i++) {
			String left2 = "#";
			String left1 = "#";
			String mid = String.valueOf(x.charAt(i));
			String right1 = "#";
			String right2 = "#";
			if (i - 2 >= 0)
				left2 = String.valueOf(x.charAt(i-2));
			if (i - 1 >= 0) 
				left1 = String.valueOf(x.charAt(i-1));
			if (i + 1 < x.length())
				right1 = String.valueOf(x.charAt(i+1));
			if (i + 2 < x.length())
				right2 = String.valueOf(x.charAt(i+2));
			String[] f = {"1"+mid, "2"+left1, "3"+right1, "4"+left2+left1,
					"5"+left1+mid, "6"+mid+right1, "7"+right1+right2};
			features.add(f);
		}
		return features;
	}
	public void updateWeights(String x, ArrayList<Integer> y, Double delta) {
		ArrayList<String[]> features = this.enumFeatures(x);
		for (int i = 0; i < features.size(); i++) {
			String[] f = features.get(i);
			for (int j = 0; j < f.length; j++) {
				this.weights.updateWeights(y.get(i)+f[j], delta);
			}
		}
		for (int i = 0; i < x.length()-1; i++) {
			this.weights.updateWeights(y.get(i) + "->"
					 + y.get(i+1), delta);
		}
	}
	
	public static boolean isSplit(String w) {
		String[] split  = {"[", "]", "@", "#", "$", "%", "&", "*", "(", ")", ".", 
				"、", "，", "。", "！", "“", "”", "》", "《", "【", "】", "；", "："};
		for (String s: split) {
			if (s.equals(w)) {
				return true;
			}
		}
		return false;
	}
	
	public ArrayList<Integer> decode(String x) {
		double[][] transitions = new double[4][4];  //转移概率
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				transitions[i][j] = this.weights.getValue(i 
						+ "->" + j, 0.0);  
			}
		}
		double[][] emissions = new double[x.length()][4];  //发射概率
		ArrayList<String[]> features = this.enumFeatures(x);
		ArrayList<Integer> res = new ArrayList<Integer>();
		for (int i = 0; i < features.size(); i++) {
			for (int j = 0; j < 4; j++) {
				double sum = 0.0;
				String[] f = features.get(i);
				for (int k = 0; k < f.length; k++) {
					sum += this.weights.getValue(j + f[k], 0.0);
				}
				emissions[i][j] = sum;
			}
		}
	
        double[][] alphas = new double[x.length()][4];
        int[][] pointers = new int[x.length()][4];
        System.arraycopy(emissions[0], 0, alphas[0], 0, 4);
        Arrays.fill(pointers[0], -1);
        for (int i = 1; i < x.length(); ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                double score = -Double.MAX_VALUE;
                for (int k = 0; k < 4; ++k)
                {
                    double s = alphas[i - 1][k] + transitions[k][j] + emissions[i][j];
                    if (s > score)
                    {
                        score = s;
                        pointers[i][j] = k;
                    }
                }
                alphas[i][j] = score;
            }
        }
        int[] tags = new int[x.length()];
        int best = -1;
        double score = -Double.MAX_VALUE;
        for (int j = 0; j < 4; ++j)
        {
            if (alphas[x.length() - 1][j] > score)
            {
                score = alphas[x.length() - 1][j];
                best = j;
            }
        }
        tags[x.length() - 1] = best;
        for (int i = x.length() - 2; i >= 0; --i)
        {
            best = pointers[i + 1][best];
            tags[i] = best;
        }
        for (int i = 0; i < tags.length; i++) {
        	res.add(tags[i]);
        }
        return res;
	}
}
