/* I declare that this code is my own work */
/* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */
import com.jogamp.opengl.GL3;
import gmaths.*;
import com.jogamp.opengl.util.texture.*;

//globe texture origin: https://www.dreamstime.com/photos-images/detailed-world-map.html
public class GlobeStand{
    private Camera camera;
    private Light light;
    private Model globe, axis, pedestal;
    private TransformNode translateXYZ, rotateGlobe;
    private Texture globeMap, pedestalMap, axisMap;
    private static final float PEDESTAL_HEIGHT = 1.0f;
    private static float ANGULAR_SPEED = 30.0f;
    private SGNode stackRoot;
    private float ROTATE_ANGLE_START = 360, ROTATE_ANGLE = ROTATE_ANGLE_START;

    public GlobeStand(GL3 gl, Vec3 initial_pos,  Camera cameraIn, Light lightIn, TextureLibrary textures){
        
        camera = cameraIn;
        light = lightIn;
        String GLOBE_NAME = "globe";
        String AXIS_NAME = "axis";
        String PEDESTAL_NAME = "pedestal";
        Mesh globeMesh = new Mesh(gl, Sphere.vertices.clone(), Sphere.indices.clone());
        Mesh pedestalMesh = new Mesh(gl, Cube.vertices.clone(), Cube.indices.clone());
        Mesh axisMesh = new Mesh(gl, Sphere.vertices.clone(), Sphere.indices.clone());
        Shader shader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_2t.txt");
        Material material = new Material(new Vec3(0.0f, 0.5f, 0.81f), new Vec3(0.0f, 0.5f, 0.81f), new Vec3(0.3f, 0.3f, 0.3f), 32.0f);
        globeMap = textures.get("globe");
        pedestalMap = textures.get("container");
        axisMap = textures.get("axis");
        Mat4 modelMatrix = Mat4Transform.translate(initial_pos.x,initial_pos.y,initial_pos.z);
        startTime = getSeconds();

        

        globe = new Model(GLOBE_NAME, globeMesh, modelMatrix, shader, material, light, camera,globeMap);
        pedestal = new Model(PEDESTAL_NAME, pedestalMesh, modelMatrix, shader, material, light, camera,pedestalMap);
        axis = new Model(AXIS_NAME, axisMesh, modelMatrix, shader, material, light, camera,axisMap);
        
        stackRoot = new NameNode("stack");
        translateXYZ = new TransformNode("translate pedestal", Mat4Transform.translate(initial_pos.x,initial_pos.y,initial_pos.z));
        
        NameNode pedestalPart = new NameNode("pedestal");
        Mat4 m =new Mat4(1);
        m = Mat4.multiply(m, Mat4Transform.translate(0,0.5f,0));
        TransformNode pedestalTransform = new TransformNode("translate(0,0.5,0)", m);
        ModelNode pedestalShape = new ModelNode("Cube(0)", pedestal);
        pedestalPart.addChild(pedestalTransform);
        pedestalTransform.addChild(pedestalShape);

        
        TransformNode translateToTopOfPedestal = new TransformNode("translate(0,"+PEDESTAL_HEIGHT+",0)",
                                                        Mat4Transform.translate(0,PEDESTAL_HEIGHT,0));

        NameNode axisPart = new NameNode("axis");
        m = Mat4.multiply(m, Mat4Transform.scale(0.2f, 4.0f, 0.2f));
        TransformNode axisTransform = new TransformNode("scale(0,4,0);translate(0,0.5,0)", m);
        ModelNode axisShape = new ModelNode("Sphere(0)", axis);
        axisPart.addChild(axisTransform);
        axisTransform.addChild(axisShape);

        TransformNode translateToTopOfAxis = new TransformNode("translate(0,"+PEDESTAL_HEIGHT+",0)",
                                                    Mat4Transform.translate(0,PEDESTAL_HEIGHT,0));



        NameNode globePart = new NameNode("globe");
        m = Mat4.multiply(m, Mat4Transform.scale(6.0f, 0.5f, 6.0f));
        m = Mat4.multiply(m, Mat4Transform.translate(0,0.5f,0));
        TransformNode globeTransform = new TransformNode("translate(0,0.5,0)", m);
        ModelNode globeShape = new ModelNode("Sphere(1)", globe);
        globePart.addChild(globeTransform);
        globeTransform.addChild(globeShape);

        rotateGlobe = new TransformNode("rotateAroundY("+ROTATE_ANGLE+")",Mat4Transform.rotateAroundY(ROTATE_ANGLE));
        
        stackRoot.addChild(translateXYZ);
        translateXYZ.addChild(pedestalPart);
        pedestalPart.addChild(translateToTopOfPedestal);
        translateToTopOfPedestal.addChild(axisPart);
        axisPart.addChild(translateToTopOfAxis);
        translateToTopOfAxis.addChild(rotateGlobe);
        rotateGlobe.addChild(globePart);
        stackRoot.update();
    }


    public void render(GL3 gl){
        globe.render(gl);
        axis.render(gl);
        pedestal.render(gl);
        updateBranches();
        stackRoot.draw(gl);
    }

    private void updateBranches() {
        double elapsedTime = getSeconds()-startTime;
        ROTATE_ANGLE = (float)(elapsedTime * ANGULAR_SPEED) % 360.0f;
        rotateGlobe.setTransform(Mat4Transform.rotateAroundY(ROTATE_ANGLE));
        stackRoot.update();

    }
  
    private double startTime;
  
    private double getSeconds() {
      return System.currentTimeMillis()/1000.0;
    }


}
