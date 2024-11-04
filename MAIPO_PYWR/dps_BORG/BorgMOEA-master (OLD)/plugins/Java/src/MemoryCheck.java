import org.borgmoea.Borg;
import org.borgmoea.ObjectiveFunction;
import org.borgmoea.Result;

public class MemoryCheck {
	
	public static final int nvars = 11;
	
	public static final int nobjs = 2;
	
	public static class DTLZ2 implements ObjectiveFunction {

		@Override
		public void evaluate(double[] vars, double[] objs, double[] constrs) {
			int i;
			int j;
			int k = nvars - nobjs + 1;
			double g = 0.0;

			for (i=nvars-k; i<nvars; i++) {
				g += Math.pow(vars[i] - 0.5, 2.0);
			}

			for (i=0; i<nobjs; i++) {
				objs[i] = 1.0 + g;

				for (j=0; j<nobjs-i-1; j++) {
					objs[i] *= Math.cos(0.5*Math.PI*vars[j]);
				}

				if (i != 0) {
					objs[i] *= Math.sin(0.5*Math.PI*vars[nobjs-i-1]);
				}
			}
		}
	}
	
	public static void main(String[] args) {
		for (int j = 0; j < 10000000; j++) {
			Borg borg = new Borg(nvars, nobjs, 0, new DTLZ2());
	
			for (int i = 0; i < nvars; i++) {
				borg.setBounds(i, 0.0, 1.0);
			}
			
			for (int i = 0; i < nobjs; i++) {
				borg.setEpsilon(i, 0.01);
			}
			
			Result result = borg.solve(1);
		}
	}

}
