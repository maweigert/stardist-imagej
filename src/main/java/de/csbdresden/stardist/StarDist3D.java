package de.csbdresden.stardist;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.menu.MenuConstants;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.NumberWidget;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

@Plugin(type = Command.class, label = "StarDist 3D", menu = {
        @Menu(label = MenuConstants.PLUGINS_LABEL, weight = MenuConstants.PLUGINS_WEIGHT, mnemonic = MenuConstants.PLUGINS_MNEMONIC),
        @Menu(label = "StarDist"),
        @Menu(label = "StarDist 3D", weight = 1)

}) 
public class StarDist3D extends StarDist3DBase implements Command {

    @Parameter
    private OpService opService;


    @Parameter(label=Opt.INPUT_IMAGE) //, autoFill=false)
    private Dataset input;

    @Parameter(label=Opt.MODEL_FILE, required=true)
    private File modelFile;

    @Parameter(label="Ray File", required=true)
    private File rayFile;

    @Parameter(label=Opt.PROB_THRESH, stepSize="0.05", min="0", max="1", style= NumberWidget.SLIDER_STYLE)
    private double probThresh = (double) Opt.getDefault(Opt.PROB_THRESH);


    @Parameter(label=Opt.NUM_TILES, min="1", stepSize="1")
    private int nTiles = (int) Opt.getDefault(Opt.NUM_TILES);

    @Parameter(label=Opt.LABEL_IMAGE, type=ItemIO.OUTPUT)
    private Dataset label;

    // ---------
    @Override
    public void run() {
        try {

            final HashMap<String, Object> paramsCNN = new HashMap<>();
            paramsCNN.put("input", input);
            paramsCNN.put("normalizeInput", true);
            paramsCNN.put("percentileBottom", 1);
            paramsCNN.put("percentileTop", 99.8);
            paramsCNN.put("clip", false);
            paramsCNN.put("nTiles", nTiles);
            paramsCNN.put("blockMultiple", 32);
            paramsCNN.put("overlap", 32);
            paramsCNN.put("batchSize", 1);
            paramsCNN.put("showProgressDialog", true);
            paramsCNN.put("modelFile", modelFile);

            status.showStatus(0,100,"Predicting probabilities and distances...");

            final Future<CommandModule> futureCNN = command.run(de.csbdresden.csbdeep.commands.GenericNetwork.class, false, paramsCNN);
            final Dataset prediction = (Dataset) futureCNN.get().getOutput("output");

            final Pair<Dataset, Dataset> probAndDist = splitPrediction(prediction);

            Pair<float[], int[]> vertsAndFaces = Utils.readRayVerticesFaces(rayFile);


            final Object[] probDistPoints =
                    filterCandidates(probAndDist, probThresh);

            float [] probArray = (float[]) probDistPoints[0];
            float [] distArray = (float[]) probDistPoints[1];
            float [] pointArray = (float[]) probDistPoints[2];

            status.showStatus(50,100,"Filtering "+probArray.length+" candidates ...");
            Pair<float[], float[]> distAndPointSurvivor = NonMaximumSuppressionSparse(probArray, distArray, pointArray,  vertsAndFaces.getA(), vertsAndFaces.getB(), (float)probThresh);

            long [] dims = new long[3];
            input.dimensions(dims);

            label = renderPolyhedra(distAndPointSurvivor.getA(), distAndPointSurvivor.getB(),
                    vertsAndFaces.getA(), vertsAndFaces.getB(), dims);

        } catch (InterruptedException | ExecutionException | IOException e) {
            e.printStackTrace();
        }
    }


    public interface LIBStardist3d extends Library {

        LIBStardist3d INSTANCE = (LIBStardist3d) Native.loadLibrary("stardist3d",
                LIBStardist3d.class);

        void _LIB_non_maximum_suppression_sparse(float[] prob, float[] dist, float[] points,
                                            int n_polys, int n_rays, int n_faces,
                                            float[] verts, int[] faces,
                                            float threshold, int use_bbox, int verbose,
                                            boolean[] results);

        void _LIB_polyhedron_to_label(float[] dist, float[] points,
                                 float[] verts, int[] faces,
                                 int n_polys, int n_rays, int n_faces,
                                 int[] labels,
                                 int nz, int ny, int nx,
                                 int render_mode, int verbose,
                                 int use_overlap_label, int overlap_label,
                                 int[] result);

    }

    private Pair<float[], float[]> NonMaximumSuppressionSparse(float[] probArray, float[] distArray, float[] pointArray,
                                             float[] vertArray, int[] faceArray, final float threshold ) {
        // this method wraps an existing c++ implementation provided as a library
        System.out.println("calling c code...");

        final int n_polys = (int)(probArray.length);
        final int n_rays = (int)(vertArray.length/3);
        final int n_faces = (int)(faceArray.length/3);

        boolean[] nms_survivors = new boolean[n_polys];

        int use_bbox = 1;
        int verbose = 1;


        System.out.println(distArray.length);
        LIBStardist3d.INSTANCE._LIB_non_maximum_suppression_sparse(probArray, distArray,  pointArray,
                n_polys, n_rays, n_faces,
                vertArray, faceArray,
                threshold, use_bbox, verbose,
                nms_survivors);

        int nSurvivors = 0;

        for (int i = 0; i < nms_survivors.length; i++)
            nSurvivors += nms_survivors[i]?1:0;

        // aggregate Survivors
        float[] distSurvivor = new float[n_rays*nSurvivors];
        float[] pointSurvivor = new float[3*nSurvivors];

        int j =0;
        for (int i = 0; i < nms_survivors.length; i++) {
            if (nms_survivors[i]) {
                System.arraycopy(pointArray, 3 * i, pointSurvivor, 3 * j, 3);
                System.arraycopy(distArray, n_rays * i, distSurvivor, n_rays * j, n_rays);
                j += 1;
            }
        }
        return new ValuePair<>(distSurvivor, pointSurvivor);
    }

    private Dataset renderPolyhedra(float[] dist, float[] points, float[] verts, int[] faces, long [] dims) {

        System.out.println(Arrays.toString(dims));

        final int n_polys = (int)(points.length/3);
        final int n_rays = (int)(verts.length/3);
        final int n_faces = (int)(faces.length/3);

        int [] result = new int[(int)(dims[0]*dims[1]*dims[2])];

        // default labels are 1...n_polys
        int[] labels = new int[n_polys];
        for (int i = 0; i < n_polys; i++)
            labels[i] = i+1;

        int render_mode=0;
        int verbose = 1;
        int use_overlap_label = 0;
        int overlap_label = 0;

        LIBStardist3d.INSTANCE._LIB_polyhedron_to_label(dist, points,verts, faces,
                n_polys, n_rays, n_faces, labels, (int)dims[2], (int)dims[1], (int)dims[0],
                render_mode, verbose, use_overlap_label, overlap_label,
                result);

        ArrayImg<UnsignedIntType, IntArray> img = ArrayImgs.unsignedInts(result, dims);

        return dataset.create(new ImgPlus(dataset.create(img)));

    }

    // get candidates as float[] arrays
    // if thats the only way to do it, I'll kill myself
    private Object[] filterCandidates(Pair<Dataset, Dataset> probAndDist, double prob_thresh) {

        final RandomAccessibleInterval<FloatType> prob = (RandomAccessibleInterval<FloatType>) probAndDist.getA().getImgPlus();
        final RandomAccessibleInterval<FloatType> dist = (RandomAccessibleInterval<FloatType>) probAndDist.getB().getImgPlus();

        final RandomAccess< FloatType > probCurs = prob.randomAccess();
        final RandomAccess< FloatType > distCurs = dist.randomAccess();

        long[] dims = new long[4];
        dist.dimensions(dims);

        ArrayList<Float> probList = new ArrayList<>();
        ArrayList<Float> distList = new ArrayList<>();
        ArrayList<Integer> pointList = new ArrayList<>();

        for (int z = 0; z < dims[2]; z++)
            for (int y = 0; y < dims[1]; y++)
                for (int x = 0; x < dims[0]; x++){
                    probCurs.setPosition(x,0);
                    probCurs.setPosition(y,1);
                    probCurs.setPosition(z,2);
                    if (probCurs.get().getRealFloat()>prob_thresh) {
                        probList.add(probCurs.get().getRealFloat());
                        pointList.add(z);
                        pointList.add(y);
                        pointList.add(x);
                        for (int j = 0; j < dims[3]; j++) {
                            distCurs.setPosition(x, 0);
                            distCurs.setPosition(y, 1);
                            distCurs.setPosition(z, 2);
                            distCurs.setPosition(j, 3);
                            distList.add(distCurs.get().getRealFloat());
                        }
                    }
                }

        float[] probArray = new float[probList.size()];
        for (int i = 0; i < probList.size(); i++)
            probArray[i]  = probList.get(i);

        float[] distArray = new float[distList.size()];
        for (int i = 0; i < distList.size(); i++)
            distArray[i]  = distList.get(i);

        float[] pointArray = new float[pointList.size()];
        for (int i = 0; i < pointList.size(); i++)
            pointArray[i]  = pointList.get(i);


        System.out.println(distArray.length);
        System.out.println(dims[3]*probArray.length);
        System.out.println(probArray.length);
        System.out.println(Arrays.toString(dims));


        if (distArray.length != dims[3]*probArray.length)
            throw new RuntimeException("inconsistent dist and prob length");

        return new Object[]{probArray,distArray,pointArray};

    }


    // this function is very cumbersome... is there a better way to do this?
    private Pair<Dataset, Dataset> splitPrediction(final Dataset prediction) {

        final RandomAccessibleInterval<FloatType> predictionRAI = (RandomAccessibleInterval<FloatType>) prediction.getImgPlus();

        long[] dims = new long[4];
        predictionRAI.dimensions(dims);

        FinalInterval interval_prob = FinalInterval.createMinMax(0,0,0,0,
                dims[0]-1,dims[1]-1,dims[2]-1,0);
        FinalInterval interval_dist = FinalInterval.createMinMax(0,0,0,1,
                dims[0]-1,dims[1]-1,dims[2]-1,dims[3]-1);

        final RandomAccessibleInterval<FloatType> probRAI = opService.transform().crop(predictionRAI, interval_prob, true);
        final RandomAccessibleInterval<FloatType> distRAI = opService.transform().crop(predictionRAI, interval_dist, true);

        final Dataset probDS = Utils.raiToDataset(dataset, Opt.PROB_IMAGE, probRAI);
        final Dataset distDS = Utils.raiToDataset(dataset, Opt.DIST_IMAGE, distRAI);

        return new ValuePair<>(probDS, distDS);
    }


    public static void main(final String... args) throws Exception {
        final ImageJ ij = new ImageJ();
        ij.launch(args);


        Dataset input = ij.scifio().datasetIO().open(StarDist3D.class.getClassLoader().getResource("img3d.tif").getFile());
        ij.ui().show(input);
        final HashMap<String, Object> params = new HashMap<>();

        File modelFile = new File(StarDist3D.class.getClassLoader().getResource("model3D.zip").getFile());
        File rayFile = new File(StarDist3D.class.getClassLoader().getResource("model_rays.json").getFile());
        params.put("modelFile", modelFile);
        params.put("rayFile", rayFile);
        params.put("nTiles", 1);
        params.put("probThresh", .8);

        // add resource folder to lib search path for testing (put libstardist.dylib in resources)
        //String resourceFolder = modelFile.getParent().toString();
        //        NativeLibrary.addSearchPath("libstardist3d.dylib", resourceFolder);
        //System.out.println(Arrays.toString((new File(resourceFolder)).list()));

        ij.command().run(StarDist3D.class, true, params);

    }

}
