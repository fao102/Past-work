/* I declare that this code is my own work */
/* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */
import gmaths.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.texture.*;


public class SurveyRobot {

    private Camera camera;
    private Light light,spotLight;
    public Vec3 position; 
    public boolean isNear = false;

    private Model sphere, cube, cube2, lightSphere;
    private Vec3 CORNER1_POS =  new Vec3(-3.45f, 0.5f, 6.5f); 
    private Vec3 CORNER2_POS = new Vec3(3.2f, 0.5f, 6.5f); 
    private Vec3 CORNER3_POS = new Vec3( 3.2f, 0.5f, -4.5f); 
    private Vec3 CORNER4_POS =  new Vec3(-3.4f, 0.5f, -4.5f); 

    private SGNode robotRoot;
    private float ROTATE_SPOTLIGHT_ANGLE = 360;
    private float ANGULAR_SPEED = 30f;
    float ROBOT_X = 0.0f; // Initial x-position
    float ROBOT_Z = 0.0f; // Initial z-position
    float SPEED = 0.1f;  // SPEED of movement
    private Vec3[] corners = {CORNER1_POS, CORNER2_POS, CORNER3_POS, CORNER4_POS};
    private int CURRENT_CORNER_INDEX = 0; // Start with the first corner
    private float CURRENT_ROTATION = 0; // Track the current rotation angle
    Vec3 dancingRobotPos;
    

    private TransformNode robotMoveTranslate, rotateBody, translateToTop, translateToTopOfPole, translateToTopOfSpotLight, translateInFrontOfBody1,translateInFrontOfBody2,rotateSpotLight;
    

    public SurveyRobot(GL3 gl, Camera cameraIn, Light lightIn, Texture texture, Texture t2, Vec3 initialPos, Vec3 robot1Pos) {
        camera = cameraIn;
        light = lightIn;
        position = initialPos;
        ROBOT_X = position.x;
        ROBOT_Z = position.z;
        dancingRobotPos = robot1Pos;
        
        sphere = makeSphere(gl, t2); // All parts use spheres for elliptical shapes.
        cube = makeCube(gl, texture);
        lightSphere = makeLightSphere(gl);
        lightSphere.material.setDiffuse(1.0f, 1.0f, 1.0f); // White light
        startTime = getSeconds();
        
        // Root structure
        robotRoot = new NameNode("root");
        robotMoveTranslate = new TransformNode("robot translate", Mat4Transform.translate(position.x, position.y, position.z));

        NameNode body = makeBody(gl, 1f, 1f, 2f, cube);
        NameNode rightEye = makeEye(gl, 0.5f, 0.25f, 0.25f, sphere, false);
        NameNode leftEye = makeEye(gl, 0.5f, 0.25f, 0.25f, sphere, true);
        NameNode pole = makePole(gl);
        NameNode spotLight = makeSpotLight(gl);
        NameNode light = new NameNode("light sphere");
        ModelNode lightSphereShape = new ModelNode("sphere(light sphere)", lightSphere);
        TransformNode lightSphereTransfrom = new TransformNode("scale light", Mat4Transform.scale(0.5f, 0.5f, 0.5f));
        light.addChild(lightSphereTransfrom);
        lightSphereTransfrom.addChild(lightSphereShape);
        double elapsedTime = getSeconds()-startTime;
        float rotateAngle = (float)(elapsedTime * ANGULAR_SPEED) % 360.0f;
        rotateBody = new TransformNode("body rotate",Mat4Transform.rotateAroundY(rotateAngle));
        translateToTop = new TransformNode("translate pole", Mat4Transform.translate(0,0.0f,0));
        translateToTopOfPole = new TransformNode("translate spotlight", Mat4Transform.translate(0,0.8f,0f));
        translateToTopOfSpotLight = new TransformNode("translate light", Mat4Transform.translate(0,0.6f,0.7f));
        translateInFrontOfBody1 = new TransformNode("translate left eye", Mat4Transform.translate(-0.25f,0.3f,1f));
        translateInFrontOfBody2 = new TransformNode("translate right eye", Mat4Transform.translate(0.25f,0.3f,1f));
        rotateSpotLight = new TransformNode("rotate spotlight", Mat4Transform.rotateAroundY(ROTATE_SPOTLIGHT_ANGLE));
        
        // Assemble the robot
        robotRoot.addChild(robotMoveTranslate); // Moves the whole robot
        robotMoveTranslate.addChild(rotateBody); // Rotates the whole robot
        rotateBody.addChild(body);
        body.addChild(translateToTop);
        body.addChild(translateInFrontOfBody1);
        body.addChild(translateInFrontOfBody2);
        translateToTop.addChild(pole);
        translateInFrontOfBody1.addChild(leftEye);
        translateInFrontOfBody2.addChild(rightEye);
        pole.addChild(translateToTopOfPole);
        translateToTopOfPole.addChild(rotateSpotLight);
        rotateSpotLight.addChild(spotLight);
        spotLight.addChild(translateToTopOfSpotLight);
        translateToTopOfSpotLight.addChild(light);
        robotRoot.update(); // Update hierarchy
    }

    private Model makeCube(GL3 gl, Texture texture) {
        String name= "cube";
        Mesh mesh = new Mesh(gl, Cube.vertices.clone(), Cube.indices.clone());
        Shader shader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_2t.txt");
        Material material = new Material(new Vec3(1.0f, 0.5f, 0.31f), new Vec3(1.0f, 0.5f, 0.31f), new Vec3(0.5f, 0.5f, 0.5f), 32.0f);
        Mat4 modelMatrix = Mat4.multiply(Mat4Transform.scale(1,1,1), Mat4Transform.translate(0,0.5f,0));
        Model cube = new Model(name, mesh, modelMatrix, shader, material, light, camera, texture, texture);
        return cube;
    }



    private Model makeSphere(GL3 gl, Texture texture) {
        String name= "sphere";
        Mesh mesh = new Mesh(gl, Sphere.vertices.clone(), Sphere.indices.clone());
        Shader shader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_1t.txt");
        Material material = new Material(new Vec3(1.0f, 0.5f, 0.31f), new Vec3(1.0f, 0.5f, 0.31f), new Vec3(0.5f, 0.5f, 0.5f), 32.0f);
        Mat4 modelMatrix = Mat4.multiply(Mat4Transform.scale(1,1,1), Mat4Transform.translate(0,0.5f,0));
        Model sphere = new Model(name, mesh, modelMatrix, shader, material, light, camera, texture);
        return sphere;
    } 

    private Model makeLightSphere(GL3 gl) {
        String name= "sphere";
        Mesh mesh = new Mesh(gl, Sphere.vertices.clone(), Sphere.indices.clone());
        Shader shader = light.shader;
        Material material =light.material;
        Mat4 modelMatrix = Mat4.multiply(Mat4Transform.scale(1,1,1), Mat4Transform.translate(0,0.5f,0));
        Model sphere = new Model(name, mesh, modelMatrix, shader, material, light, camera);
        return sphere;
    } 
    
    private NameNode makeBody(GL3 gl, float X_SCALE, float Y_SCALE, float Z_SCALE, Model cube) {
        NameNode base = new NameNode("base");
        Mat4 m = new Mat4(1);
        m = Mat4.multiply(m, Mat4Transform.scale(X_SCALE, Y_SCALE, Z_SCALE)); // Scale for a flat base
        TransformNode baseTransform = new TransformNode("base transform", m);
        ModelNode baseShape = new ModelNode("Cube(base)", cube);
        base.addChild(baseTransform);
        baseTransform.addChild(baseShape);
        return base;
    }
    private NameNode makeEye(GL3 gl, float X_SCALE,float Y_SCALE, float Z_SCALE, Model sphere, boolean isLeft) {
        NameNode eye = new NameNode(isLeft ? "left eye" : "right eye");
        Mat4 m = Mat4Transform.scale(X_SCALE, Y_SCALE, Z_SCALE);
        TransformNode eyeTransform = new TransformNode("eye transformation", m);
        ModelNode eyeShape = new ModelNode("Sphere(eye)", sphere);
        eye.addChild(eyeTransform);
        eyeTransform.addChild(eyeShape);
        return eye;
    }

    private NameNode makePole(GL3 gl){
        NameNode pole = new NameNode("pole");
        Mat4 m =new Mat4(1);
        m = Mat4.multiply(m, Mat4Transform.translate(0,0.5f,0));
        m = Mat4.multiply(m, Mat4Transform.scale(0.2f, 1.5f, 0.2f));
        TransformNode poleTransform = new TransformNode("scale(0,4,0);translate(0,0.5,0)", m);
        ModelNode poleShape = new ModelNode("Sphere(pole)", sphere);
        pole.addChild(poleTransform);
        poleTransform.addChild(poleShape);
        return pole;
    }

    private NameNode makeSpotLight(GL3 gl){
        NameNode pole = new NameNode("spotlight");
        Mat4 m =new Mat4(1);
        m = Mat4.multiply(m, Mat4Transform.translate(0,0.5f,0));
        m = Mat4.multiply(m, Mat4Transform.scale(0.5f, 0.5f, 1f));
        TransformNode poleTransform = new TransformNode("scale(0,4,0);translate(0,0.5,0)", m);
        ModelNode poleShape = new ModelNode("Sphere(spotlight)", sphere);
        pole.addChild(poleTransform);
        poleTransform.addChild(poleShape);
        return pole;
    }


    public void updateSpotlight() {
        double elapsedTime = getSeconds()-startTime;
        float rotateSpotLightAngle = (float)(elapsedTime * ANGULAR_SPEED) % ROTATE_SPOTLIGHT_ANGLE;
        rotateSpotLight.setTransform(Mat4Transform.rotateAroundY(rotateSpotLightAngle));
        rotateSpotLight.update();
    }



    
    public void updateRobotPosition() {
        updateSpotlight();
        float CORNER_RANGE = 0.2f; // CORNER_RANGE within which the robot is considered to have reached the corner
        float ROBOT_RANGE = 5f;
        Vec3 targetCorner = corners[CURRENT_CORNER_INDEX]; // Current target corner
    
        // Calculate direction vector toward the target corner
        float dx = targetCorner.x - ROBOT_X;
        float dz = targetCorner.z - ROBOT_Z;
        float distanceToCorner = (float) Math.sqrt(dx * dx + dz * dz);

        float robot1Dx = dancingRobotPos.x - ROBOT_X;
        float robot1Dz = dancingRobotPos.z - ROBOT_Z;
        float distanceToRobot = (float) Math.sqrt(robot1Dx * robot1Dx + robot1Dz * robot1Dz);

        if (distanceToRobot <= ROBOT_RANGE){
            isNear = true;
        }else{
            isNear = false;
        }
        if (distanceToCorner <= CORNER_RANGE) {
            // Robot reached the corner, perform a 90-degree turn
            SPEED = 0; // Stop the robot
            CURRENT_ROTATION += 90f; // Increment the rotation angle by 90 degrees
            rotateBody.setTransform(Mat4Transform.rotateAroundY(CURRENT_ROTATION)); // Rotate the robot
            rotateBody.update();

            if(CURRENT_CORNER_INDEX != 3){
                CURRENT_CORNER_INDEX = (CURRENT_CORNER_INDEX + 1) ;
            }else{
                CURRENT_CORNER_INDEX =0;
            }
            
        }
        SPEED = 0.1f;
    
        // Normalize the direction vector to maintain consistent SPEED
        float directionMagnitude = (float) Math.sqrt(dx * dx + dz * dz);
        float normalizedDx = dx / directionMagnitude;
        float normalizedDz = dz / directionMagnitude;
    
        // Update position based on direction and SPEED
        float moveX = normalizedDx * SPEED;
        float moveZ = normalizedDz * SPEED;
    
        ROBOT_X += moveX;
        ROBOT_Z += moveZ;
    
        // Update the robot's global translation
        robotMoveTranslate.setTransform(Mat4Transform.translate(ROBOT_X, 0, ROBOT_Z));
        robotMoveTranslate.update();
    
        position = new Vec3(ROBOT_X, position.y, ROBOT_Z); // Update position for tracking
       
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