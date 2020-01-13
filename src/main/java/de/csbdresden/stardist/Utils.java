package de.csbdresden.stardist;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Stream;

import de.lighti.clipper.Path;
import de.lighti.clipper.Point.LongPoint;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImgPlus;
import net.imagej.axis.AxisType;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelingMapping;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import org.json.JSONArray;
import org.json.JSONObject;

public class Utils {

    public static PolygonRoi toPolygonRoi(Path poly) {
        int n = poly.size();
        int[] x = new int[n];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            LongPoint p = poly.get(i);
            x[i] = (int) p.getX();
            y[i] = (int) p.getY();
        }
        return new PolygonRoi(x,y,n,Roi.POLYGON);
    }

    public static double[] rayAngles(int n) {
        double[] angles = new double[n];
        double st = (2*Math.PI)/n;
        for (int i = 0; i < n; i++) angles[i] = st*i;
        return angles;
    }

    public static List<Integer> argsortDescending(final List<Float> list) {
        Integer[] indices = new Integer[list.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return -Float.compare(list.get(i), list.get(j));
            }
        });
        return Arrays.asList(indices);
    }

    public static LinkedHashSet<AxisType> orderedAxesSet(Dataset image) {
        final int numDims = image.numDimensions();
        final LinkedHashSet<AxisType> axes = new LinkedHashSet<>(numDims);
        for (int d = 0; d < numDims; d++)
            axes.add(image.axis(d).type());
        return axes;
    }
    
    public static Dataset raiToDataset(final DatasetService dataset, final String name, final RandomAccessibleInterval rai, final AxisType... axes) {
        // is there a better way?
        // https://forum.image.sc/t/convert-randomaccessibleinterval-to-imgplus-or-dataset/8535/6        
        return dataset.create(new ImgPlus(dataset.create(rai), name, axes));
    }
    
    public static Dataset raiToDataset(final DatasetService dataset, final String name, final RandomAccessibleInterval rai, final Stream<AxisType> axesStream) {
        return raiToDataset(dataset, name, rai, axesStream.toArray(AxisType[]::new));
    }

    public static Dataset raiToDataset(final DatasetService dataset, final String name, final RandomAccessibleInterval rai, final Collection<AxisType> axesCollection) {
        return raiToDataset(dataset, name, rai, axesCollection.stream());
    }

    public static Pair<float[], int[]> readRayVerticesFaces(final File raysFile) throws IOException {

        final String content = new String ( Files.readAllBytes(raysFile.toPath()));
        System.out.println(content);

        JSONObject jo = new JSONObject(content);

        JSONArray ja_vert = jo.getJSONArray("vertices");
        JSONArray ja_face = jo.getJSONArray("faces");

        if (ja_vert.length()==0 || ja_face.length()==0)
            throw new RuntimeException("not vertices and faces found");

        float[] vertices = new float[ja_vert.length()];
        for(int i=0;i<ja_vert.length();i++)
            vertices[i] = (float)ja_vert.getDouble(i);
        int[] faces = new int[ja_face.length()];
        for(int i=0;i<ja_face.length();i++)
            faces[i] = ja_face.getInt(i);

        if (ja_vert.length()%3!=0 || ja_face.length()%3!=0)
            throw new RuntimeException("vertices and faces should be divisible by 3!");

        System.out.println("found "+vertices.length/3+" vertices and "+ faces.length/3 +" faces");
        return new ValuePair<float[], int[]>(vertices, faces);

    }

    public static ImgLabeling< Integer, IntType> convertImgToLabelImage(Img< IntType > label_img )
    {
        final ImgLabeling< Integer, IntType > imgLabeling = new ImgLabeling<>( label_img );

        int maximumLabel = -1;
        Cursor<IntType> curs = label_img.cursor();
        // iterate over the input
        while ( curs.hasNext()) {
            curs.fwd();
            maximumLabel = ((int)curs.get().getInteger()>maximumLabel)?(int)curs.get().getInteger():maximumLabel;
        }
        System.out.println("Maximum value of label image: " + maximumLabel);


        final ArrayList<Set< Integer >> labelSets = new ArrayList< >();

        labelSets.add( new HashSet<>() ); // empty 0 label
        for ( int label = 1; label <= maximumLabel; ++label )
        {
            final HashSet< Integer > set = new HashSet< >();
            set.add( label );
            labelSets.add( set );
        }

        new LabelingMapping.SerialisationAccess< Integer >( imgLabeling.getMapping() )
        {
            {
                super.setLabelSets( labelSets );
            }
        };

        return imgLabeling;
    }

}
