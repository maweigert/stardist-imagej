package de.csbdresden.stardist;

import ij.ImagePlus;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.DialogPrompt.MessageType;
import org.scijava.ui.UIService;

import java.net.URL;

public abstract class StarDist3DBase {
    
    @Parameter
    protected LogService log;

    @Parameter
    protected UIService ui;

    @Parameter
    protected CommandService command;

    @Parameter
    protected DatasetService dataset;
    
    @Parameter
    protected StatusService status;
    
    // ---------
    
    protected ImagePlus labelImage = null;
    protected int labelId = 0;
    protected long labelCount = 0;
    protected static final int MAX_LABEL_ID = 65535;
    
    // ---------
    
    protected URL getResource(final String name) {
        return StarDist3DBase.class.getClassLoader().getResource(name);
    }        
    
    protected boolean showError(String msg) {
        ui.showDialog(msg, MessageType.ERROR_MESSAGE);
        // log.error(msg);
        return false;
    }
    
    // ---------

}
