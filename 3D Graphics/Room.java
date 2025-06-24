/* I declare that this code is my own work */
/* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */

import gmaths.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.texture.*;


 /**
 * This class stores the Floor
 *

 */

public class Room {

  private Camera camera;
  private Light light;

  private Model floor, rightWall, leftWall, backWall, ceiling, stars, stars2, stars3;
  private Texture backgroundMap, cloudMap, jadeMap, diffuseFile, specularFile, marbleMap;
  private Mat4[] roomTransforms;
  private static final float WALL_WIDTH = 8f;    // Width of the wall
  private static final float WALL_HEIGHT = 16f; // Height of the wall
  private static final float WINDOW_WIDTH = 4f;  // Width of the window
  private static final float WINDOW_HEIGHT = 4f; // Height of the window
  private Vec3 ambientLightColor;
 
   
  public Room(GL3 gl, float [] room_dims, Camera cameraIn, Light lightIn, TextureLibrary textures) {

    camera = cameraIn;
    light = lightIn;
    ambientLightColor = new Vec3(0.2f, 0.2f, 0.2f);
    
    //left wall vertices
    float[] LW_VERTICES = {
      // Bottom part of the wall
      -0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // bottom-left
      -0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f,  // bottom-right
       0.5f,  0.0f,  0.5f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // top-right
       0.5f,  0.0f, -0.5f,  0.0f, 1.0f, 0.0f,  1.0f, 1.0f,  // top-left
  
      // Extra vertices for the hole
      -0.2f,  0.0f, -0.2f,  0.0f, 1.0f, 0.0f,  0.3f, 0.7f,  // inner-bottom-left
      -0.2f,  0.0f,  0.2f,  0.0f, 1.0f, 0.0f,  0.3f, 0.3f,  // inner-bottom-right
       0.2f,  0.0f,  0.2f,  0.0f, 1.0f, 0.0f,  0.7f, 0.3f,  // inner-top-right
       0.2f,  0.0f, -0.2f,  0.0f, 1.0f, 0.0f,  0.7f, 0.7f   // inner-top-left
    };


    float[] RW_VERTICES = {      // position, colour, tex coords
      -0.5f, 0.0f, -0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 2.0f,  // top left
      -0.5f, 0.0f,  0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f,  // bottom left
       0.5f, 0.0f,  0.5f,  0.0f, 1.0f, 0.0f,  2.0f, 0.0f,  // bottom right
       0.5f, 0.0f, -0.5f,  0.0f, 1.0f, 0.0f,  2.0f, 2.0f   // top right
    };
  
    int[] LW_INDICES = {
      // Bottom rectangle (below the hole)
      0, 1, 5,   0, 5, 4,
      
      // Top rectangle (above the hole)
      7, 6, 2,   7, 2, 3,
      
      // Left rectangle (to the left of the hole)
      0, 4, 7,   0, 7, 3,
      
      // Right rectangle (to the right of the hole)
      5, 1, 2,   5, 2, 6
    };

    String FLOOR_NAME = "floor";
    String LW_NAME = "right wall";
    String RW_NAME = "left wall";
    String BW_NAME = "back wall";
    String CEILING_NAME = "ceiling";
    String STARS_NAME = "stars";
    Mesh mesh = new Mesh(gl, TwoTriangles.vertices.clone(), TwoTriangles.indices.clone());
    Mesh rightWallMesh = new Mesh(gl, RW_VERTICES,TwoTriangles.indices);
    Mesh leftWallMesh = new Mesh(gl, LW_VERTICES, LW_INDICES);
    Shader shader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_2t.txt");
    Shader bwShader = new Shader(gl, "assets/shaders/vs_standard.txt", "assets/shaders/fs_standard_2t_test.txt");
    Material material = new Material(new Vec3(0.0f, 0.5f, 0.81f), new Vec3(0.0f, 0.5f, 0.81f), new Vec3(0.3f, 0.3f, 0.3f), 32.0f);
    Material backWallMaterial = new Material(
        new Vec3(0.1f, 0.1f, 0.1f),
        new Vec3(1.0f, 1.0f, 1.0f), 
        new Vec3(2.0f, 2.0f, 2.0f), 
        100.0f                       
    );
    diffuseFile = textures.get("diffuse_file");
    specularFile = textures.get("specular_file");
    jadeMap = textures.get("jade");
    backgroundMap = textures.get("chequerboard");
    marbleMap = textures.get("marble2");
    cloudMap = textures.get("cloud");
    cloudMap.setTexParameteri(gl, GL3.GL_TEXTURE_MIN_FILTER, GL3.GL_LINEAR);
    cloudMap.setTexParameteri(gl, GL3.GL_TEXTURE_MAG_FILTER, GL3.GL_LINEAR);
    cloudMap.setTexParameteri(gl, GL3.GL_TEXTURE_WRAP_S, GL3.GL_REPEAT);
    cloudMap.setTexParameteri(gl, GL3.GL_TEXTURE_WRAP_T, GL3.GL_REPEAT);

    Mat4 modelMatrix = Mat4Transform.scale(room_dims[0],1f,room_dims[1]);


    floor = new Model(FLOOR_NAME, mesh, modelMatrix, shader, material, light, camera, jadeMap);
    rightWall = new Model(RW_NAME, rightWallMesh, modelMatrix, shader, material, light, camera, cloudMap);
    leftWall = new Model(LW_NAME, leftWallMesh, modelMatrix, shader, material, light, camera, marbleMap);
    backWall = new Model(
      BW_NAME,
      mesh, 
      modelMatrix,
      bwShader,
      backWallMaterial,
      light, 
      camera,
      backgroundMap
    );
    backWall.setDiffuse(diffuseFile);
    backWall.setSpecular(specularFile);

    ceiling = new Model(CEILING_NAME, mesh, modelMatrix, shader, material, light, camera, backgroundMap);
    stars = new Model(STARS_NAME, mesh, modelMatrix, shader, material, light, camera, textures.get("stars"));
    stars2 = new Model(STARS_NAME, mesh, modelMatrix, shader, material, light, camera, textures.get("stars"));
    stars3 = new Model(STARS_NAME, mesh, modelMatrix, shader, material, light, camera, textures.get("stars"));
    shader.setVec3(gl, "ambientLightColor", ambientLightColor);
    

    roomTransforms = setupRoomTransforms();
  }

  public void render(GL3 gl) {
    floor.setModelMatrix(getMforFloor());       
    floor.render(gl);
    rightWall.setModelMatrix(getMforRightWall());       
    rightWall.render(gl);
    leftWall.setModelMatrix(getMforLeftWall());   
    leftWall.render(gl);  
    backWall.setModelMatrix(getMforBackWall());
    backWall.render(gl);
    ceiling.setModelMatrix(getMforCeiling());
    ceiling.render(gl);
    stars.setModelMatrix(getMforStars());
    stars.render(gl);
    stars2.setModelMatrix(getMforStars2());
    stars2.render(gl);
    stars3.setModelMatrix(getMforStars3());
    stars3.render(gl);
    

  }

  //room transforms
  private Mat4[] setupRoomTransforms() {
    Mat4[] t = new Mat4[8];
    t[0] = getMforFloor();
    t[1] = getMforBackWall();
    t[2] = getMforLeftWall();
    t[3] = getMforRightWall();
    t[4] = getMforCeiling();
    t[5] = getMforStars();
    t[6] = getMforStars2();
    t[7] = getMforStars3();


    return t;
  }

  
  private Mat4 getMforFloor() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH, 1f, WALL_HEIGHT), modelMatrix);
    return modelMatrix;
  }

  
  private Mat4 getMforBackWall() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH,1f,WALL_WIDTH), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundX(90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(0,WALL_WIDTH*0.5f,-WALL_HEIGHT*0.5f), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforLeftWall() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH,1f,WALL_HEIGHT), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundZ(-90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(-WALL_WIDTH*0.5f,WALL_WIDTH*0.5f,0), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforRightWall() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH,1f,WALL_HEIGHT), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundZ(90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(WALL_WIDTH*0.5f,WALL_WIDTH*0.5f,0), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforCeiling() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH,1f,WALL_HEIGHT), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundZ(180), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(0,WALL_WIDTH,0), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforStars() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH+1,1f,WALL_HEIGHT), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundZ(-90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(-WALL_WIDTH*0.55f,WALL_WIDTH*0.5f,0), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforStars2() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH+1,1f,WALL_WIDTH), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundX(90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(0,WALL_WIDTH*0.55f,-WALL_HEIGHT*0.5f), modelMatrix);
    return modelMatrix;
  }

  private Mat4 getMforStars3() {
    Mat4 modelMatrix = new Mat4(1);
    modelMatrix = Mat4.multiply(Mat4Transform.scale(WALL_WIDTH+1,1f,WALL_HEIGHT), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.rotateAroundZ(90), modelMatrix);
    modelMatrix = Mat4.multiply(Mat4Transform.translate(WALL_WIDTH*0.55f,WALL_WIDTH*0.5f,0), modelMatrix);
    return modelMatrix;
  }




}