/* I declare that this code is my own work */
/* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */
import gmaths.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.texture.*;

 /**
 * This class stores the Robot
 *
 * @author    Dr Steve Maddock
 * @version   1.0 (31/08/2022)
 */

public class DancingRobot {

  private Camera camera;
  private Light light;

  private Model sphere, sphere2, cube, cube2;

  private SGNode robotRoot;
  public Vec3 position; 
 
  private TransformNode robotMoveTranslate, rotateBody, rotateRArm,rotateLArm ;
   
  public DancingRobot(GL3 gl, Camera cameraIn, Light lightIn, Texture t1, Texture t2, Vec3 initial_pos) {
    this.camera = camera = cameraIn;
    this.light= light = lightIn;
    position = new Vec3(initial_pos);

    sphere = makeSphere(gl, t1); // All parts use spheres for elliptical shapes.
    sphere2 = makeSphere(gl, t2);

    // Dimensions based on design
    float EYE_SCALE = 0.3f;
    float EYE_OFFSET = 0.4f; // Distance from the center of the head
    float ANTENNA_HEIGHT = 2.5f;
    float ANTENNA_SCALE = 0.2f;
    float ANTENNA_ORIENTATION1 = 0;
    float ANTENNA_ORIENTATION2 = -20;
    float ANTENNA_ORIENTATION3 = 20;
    float LEG_LENGTH = 0.5f;
    float LEG_SCALE = 0.75f;
    float LEG_HEIGHT = LEG_LENGTH+0.5f;

    float BODY_HEIGHT = 1f;
    float BODY_WIDTH = 0.9f;
    float BODY_DEPTH = 1f;
    float ARM_LENGTH = 1.0f;
    float ARM_SCALE = 0.5f;

    float HEAD_WIDTH = 1f;
    float HEAD_HEIGHT = 1f;
    float HEAD_DEPTH= 1f;
    float BASE_SCALE = 1f;
    float BASE_HEIGHT = 0.5f;
    

    // Root structure
    robotRoot = new NameNode("root");
    robotMoveTranslate = new TransformNode("robot translate", Mat4Transform.translate(initial_pos.x, initial_pos.y, initial_pos.z));

    // Parts
    NameNode body = makeEllipticalBody(gl, LEG_LENGTH, BODY_WIDTH, BODY_HEIGHT, BODY_DEPTH, sphere);
    NameNode head = makeHead(gl, BASE_HEIGHT,LEG_HEIGHT, BODY_HEIGHT, HEAD_WIDTH, HEAD_HEIGHT, HEAD_DEPTH, sphere);
    NameNode leftArm = makeEllipticalArm(gl, LEG_HEIGHT,BODY_WIDTH, BODY_HEIGHT, ARM_LENGTH, ARM_SCALE, sphere, true);
    NameNode rightArm = makeEllipticalArm(gl, LEG_HEIGHT, BODY_WIDTH, BODY_HEIGHT, ARM_LENGTH, ARM_SCALE, sphere, false);
    NameNode leg1 = makeEllipticalLeg(gl,BODY_WIDTH, BASE_HEIGHT, LEG_HEIGHT, LEG_SCALE, sphere);
    NameNode leg2 = makeEllipticalLeg2(gl,BODY_WIDTH, BASE_HEIGHT, LEG_HEIGHT, LEG_SCALE, sphere2);
    NameNode base = makeBase(gl, BASE_SCALE, BASE_HEIGHT, sphere);
    NameNode eye = makeEye(gl, HEAD_HEIGHT, EYE_SCALE, EYE_OFFSET, sphere2);
    NameNode antenna = makeAntenna(gl, HEAD_HEIGHT, ANTENNA_HEIGHT, ANTENNA_SCALE, sphere, ANTENNA_ORIENTATION1);
    NameNode antenna2 = makeAntenna(gl, HEAD_HEIGHT, ANTENNA_HEIGHT, ANTENNA_SCALE, sphere,ANTENNA_ORIENTATION2);
    NameNode antenna3 = makeAntenna(gl, HEAD_HEIGHT, ANTENNA_HEIGHT, ANTENNA_SCALE, sphere,ANTENNA_ORIENTATION3);


    TransformNode translateToTopOfBase = new TransformNode("translate leg1",
                                                        Mat4Transform.translate(0,BASE_HEIGHT,0));

    TransformNode translateToTopOfLeg1 = new TransformNode("translate leg2",
                                                        Mat4Transform.translate(0,LEG_HEIGHT-0.5f,0));

    TransformNode translateToTopOfLeg2 = new TransformNode("translate body",
                                                        Mat4Transform.translate(0,LEG_HEIGHT-0.5f,0));

    TransformNode translateToTopOfBody = new TransformNode("translate head",
                                                        Mat4Transform.translate(0,BODY_HEIGHT-0.5f,0));
    


    TransformNode translateToTopOfHead = new TransformNode("translate antenna",
                                                        Mat4Transform.translate(0,BODY_HEIGHT-0.5f,0));

    TransformNode translateToFrontOfHead = new TransformNode("translate eye",
                                                        Mat4Transform.translate(0,BODY_HEIGHT-1f,EYE_OFFSET));
    
 
    
    TransformNode rotateToRight = new TransformNode("rotate right arm",
                                                        Mat4Transform.rotateAroundZ(-40));
    
   
    
    
    
    TransformNode rotateToLeft = new TransformNode("rotate left arm",
                                                        Mat4Transform.rotateAroundZ(40));
                                                      
                                                        
    

    rotateBody = new TransformNode("body rotate",Mat4Transform.rotateAroundZ(40));
    rotateLArm = new TransformNode("arm rotate",Mat4Transform.rotateAroundZ(0));
    rotateRArm = new TransformNode("arm rotate",Mat4Transform.rotateAroundZ(0));

    
    // Assemble the robot
    robotRoot.addChild(robotMoveTranslate);
    robotMoveTranslate.addChild(base);
    base.addChild(translateToTopOfBase);
    translateToTopOfBase.addChild(leg1);
    leg1.addChild(translateToTopOfLeg1);
    translateToTopOfLeg1.addChild(leg2);
    leg2.addChild(translateToTopOfLeg2);
    translateToTopOfLeg2.addChild(rotateBody);
    rotateBody.addChild(body);
    body.addChild(translateToTopOfBody);
    body.addChild(rotateToLeft);
    body.addChild(rotateToRight);
    rotateToRight.addChild(rotateRArm);
    rotateToLeft.addChild(rotateLArm);
    rotateRArm.addChild(rightArm);
    rotateLArm.addChild(leftArm);
    translateToTopOfBody.addChild(head);
    head.addChild(translateToFrontOfHead);
    head.addChild(translateToTopOfHead);
    translateToFrontOfHead.addChild(eye);
    translateToTopOfHead.addChild(antenna);
    translateToTopOfHead.addChild(antenna2);
    translateToTopOfHead.addChild(antenna3);



  

      
    robotRoot.update(); // Update hierarchy
  }

  private NameNode makeEllipticalBody(GL3 gl, float LEG_HEIGHT,float BODY_WIDTH, float BODY_HEIGHT, float BODY_DEPTH, Model sphere) {
    NameNode body = new NameNode("body");
    Mat4 m = Mat4Transform.scale(BODY_WIDTH, BODY_HEIGHT, BODY_DEPTH);
    TransformNode bodyTranslate= new TransformNode("body transform", m);
    ModelNode bodyShape = new ModelNode("Sphere(body)", sphere);
    body.addChild(bodyTranslate);
    bodyTranslate.addChild(bodyShape);
    return body;
  }

  private NameNode makeEllipticalArm(GL3 gl, float LEG_HEIGHT,float BODY_WIDTH, float BODY_HEIGHT,float ARM_LENGTH, float ARM_SCALE, Model sphere, boolean isLeft) {
      NameNode arm = new NameNode(isLeft ? "left arm" : "right arm");
      Mat4 m = Mat4Transform.scale(ARM_SCALE, ARM_LENGTH, ARM_SCALE);
      float direction = isLeft ? 1 : -1;
      m = Mat4.multiply(Mat4Transform.translate(direction * ((BODY_WIDTH*0.1f) + ARM_SCALE), LEG_HEIGHT-1.5f, 0),m);
      TransformNode armTransform = new TransformNode("arm transformation", m);
      ModelNode armShape = new ModelNode("Sphere(arm)", sphere);
      arm.addChild(armTransform);
      armTransform.addChild(armShape);
      return arm;
  }

  private NameNode makeBase(GL3 gl, float BASE_SCALE, float BASE_HEIGHT, Model sphere) {
    NameNode base = new NameNode("base");
    Mat4 m = new Mat4(1);
    m = Mat4.multiply(m, Mat4Transform.scale(BASE_SCALE, BASE_HEIGHT, BASE_SCALE)); // Scale for a flat base
    TransformNode baseTransform = new TransformNode("base transform", m);
    ModelNode baseShape = new ModelNode("Sphere(base)", sphere);
    base.addChild(baseTransform);
    baseTransform.addChild(baseShape);
    return base;
  }

  private NameNode makeHead(GL3 gl, float BASE_HEIGHT, float LEG_HEIGHT,float BODY_HEIGHT, float HEAD_WIDTH, float HEAD_HEIGHT, float HEAD_DEPTH, Model sphere) {
    NameNode head = new NameNode("head"); 
    Mat4 m = new Mat4(1);
    m = Mat4.multiply(m, Mat4Transform.scale(HEAD_WIDTH,HEAD_HEIGHT,HEAD_DEPTH));
    TransformNode headTransform = new TransformNode("head transform", m);
    ModelNode headShape = new ModelNode("Sphere(head)", sphere);
    head.addChild(headTransform);
    headTransform.addChild(headShape);
    return head;
  }

  private NameNode makeEllipticalLeg(GL3 gl, float BODY_WIDTH, float BASE_HEIGHT, float LEG_LENGTH, float LEG_SCALE, Model sphere) {
      NameNode leg = new NameNode("leg");
      Mat4 m = Mat4Transform.scale(LEG_SCALE, LEG_LENGTH, LEG_SCALE);
      TransformNode legTransform = new TransformNode("leg transform", m);
      ModelNode legShape = new ModelNode("Sphere(leg)", sphere);
      leg.addChild(legTransform);
      legTransform.addChild(legShape);
      return leg;
  }

  private NameNode makeEllipticalLeg2(GL3 gl, float BODY_WIDTH, float BASE_HEIGHT, float LEG_LENGTH, float LEG_SCALE, Model sphere) {
    NameNode leg = new NameNode("leg");
    Mat4 m = Mat4Transform.scale(LEG_SCALE*1.2f, LEG_LENGTH, LEG_SCALE);
    TransformNode legTransform = new TransformNode("leg transform", m);
    ModelNode legShape = new ModelNode("Sphere(leg)", sphere);
    leg.addChild(legTransform);
    legTransform.addChild(legShape);
    return leg;
  }


  private Model makeSphere(GL3 gl, Texture t1) {
    String name= "sphere";
    Mesh mesh = new Mesh(gl, Sphere.vertices.clone(), Sphere.indices.clone());
    Shader shader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_1t.txt");
    Material material = new Material(new Vec3(1.0f, 0.5f, 0.31f), new Vec3(1.0f, 0.5f, 0.31f), new Vec3(0.5f, 0.5f, 0.5f), 32.0f);
    Mat4 modelMatrix = Mat4.multiply(Mat4Transform.scale(4,4,4), Mat4Transform.translate(0,0.5f,0));
    Model sphere = new Model(name, mesh, modelMatrix, shader, material, light, camera, t1);
    return sphere;
  } 

  private NameNode makeEye(GL3 gl, float HEAD_HEIGHT, float EYE_SCALE, float EYE_OFFSET, Model sphere) {
    NameNode eyeNode = new NameNode("eye");
    Mat4 m = new Mat4(1);
    m = Mat4.multiply(Mat4Transform.scale(EYE_SCALE, EYE_SCALE, EYE_SCALE),m); 
    TransformNode eyeTransform = new TransformNode("eye transform", m);
    ModelNode eyeShape = new ModelNode("Sphere(eye)", sphere);
    eyeNode.addChild(eyeTransform);
    eyeTransform.addChild(eyeShape);
    return eyeNode;
  }

  private NameNode makeAntenna(GL3 gl, float HEAD_WIDTH, float ANTENNA_HEIGHT, float ANTENNA_SCALE, Model sphere, float orientation) {
    NameNode antenna = new NameNode("antenna");
    Mat4 m = new Mat4(1);
    m = Mat4.multiply(Mat4Transform.scale(0.3f, 1, 0.5f), m); // Scale to make it a tall shape
    m =  Mat4.multiply(Mat4Transform.rotateAroundZ(orientation),m); // Scale to make it a tall shape
    if (orientation != 0){
      if (orientation<0){
        m = Mat4.multiply(Mat4Transform.translate(HEAD_WIDTH/2, 0, 0),m);
      }else{
        m = Mat4.multiply(Mat4Transform.translate(-HEAD_WIDTH/2, 0, 0),m);
      }
      
    }
    TransformNode antennaTransform = new TransformNode("antenna transform", m);
    ModelNode antennaShape = new ModelNode("Sphere(antenna)", sphere);
    antenna.addChild(antennaTransform);
    antennaTransform.addChild(antennaShape);
    return antenna;
  }

  public void updateAnimation() {
    double elapsedTime = startTime - getSeconds();
    float rotateBodyAngle = 40f*(float)Math.sin(elapsedTime);
    float rotateRArmAngle = 20f*(float)Math.sin(elapsedTime);
    float rotateLArmAngle = 20f*(float)Math.sin(elapsedTime);
    rotateBody.setTransform(Mat4Transform.rotateAroundZ(rotateBodyAngle));
    rotateLArm.setTransform(Mat4Transform.rotateAroundZ(rotateRArmAngle));
    rotateRArm.setTransform(Mat4Transform.rotateAroundZ(rotateLArmAngle));
    rotateBody.update();
    rotateLArm.update();
    rotateRArm.update();
  }

  private double startTime;
  
  private double getSeconds() {
    return System.currentTimeMillis()/1000.0;
  }


  public void render(GL3 gl) {
    robotRoot.draw(gl);
  }

  public void dispose(GL3 gl) {
    sphere.dispose(gl);
    cube.dispose(gl);
    cube2.dispose(gl);
  }
}