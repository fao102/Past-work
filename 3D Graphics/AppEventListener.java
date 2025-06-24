
import gmaths.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.texture.*;


public class AppEventListener implements GLEventListener {
  private Camera camera;
    
  
  /* The constructor is not used to initialise anything */
  public AppEventListener(Camera camera) {
    this.camera = camera;
    this.camera.setPosition(new Vec3(-1f,7f,14f));
  }
  
  
  // ***************************************************
  /*
   * METHODS DEFINED BY GLEventListener
   */
  /* Initialisation */
  public void init(GLAutoDrawable drawable) {   
    GL3 gl = drawable.getGL().getGL3();
                                    // Retrieve the gl context.
    System.err.println("Chosen GLCapabilities: " + drawable.getChosenGLCapabilities());
                                    // Print some diagnostic info.
                                    // Useful, as it shows something is happening.
    System.err.println("INIT GL IS: " + gl.getClass().getName());
    System.err.println("GL_VENDOR: " + gl.glGetString(GL.GL_VENDOR));
    System.err.println("GL_RENDERER: " + gl.glGetString(GL.GL_RENDERER));
    System.err.println("GL_VERSION: " + gl.glGetString(GL.GL_VERSION));
    gl.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
                                    // Set the background colour for the OpenGL context: 0.0f <= {r,g,b,alpha} <= 1.0f.
    gl.glClearDepth(1.0f); 
        // Required for z buffer work in later examples. Will be explained there.
    gl.glEnable(GL.GL_DEPTH_TEST);  // Required for z buffer work in later examples.
    gl.glDepthFunc(GL.GL_LESS);     // Required for z buffer work in later examples.
    // gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL3.GL_LINE); 
    gl.glFrontFace(GL.GL_CCW);    // default is 'CCW'
    gl.glEnable(GL.GL_CULL_FACE); // default is 'not enabled'
    gl.glCullFace(GL.GL_BACK);   // default is 'back', assuming CCW
    createRandomNumbers();
    initialise(gl);
    startTime = getSeconds();
  }
  
  /* Called to indicate the drawing surface has been moved and/or resized  */
  public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
    GL3 gl = drawable.getGL().getGL3();
    gl.glViewport(x, y, width, height);
    float aspect = (float)width/(float)height;
    Mat4 perspective = Mat4Transform.perspective(45, aspect);
    camera.setPerspectiveMatrix(perspective);
  }
  /* Draw */
  public void display(GLAutoDrawable drawable) {
    GL3 gl = drawable.getGL().getGL3();
    render(gl);                     // A separate method is used for rendering the scene.
                                    // This reduces the clutter in this method.
  }
  /* Clean up memory, if necessary */
  public void dispose(GLAutoDrawable drawable) {
    GL3 gl = drawable.getGL().getGL3();
    light.dispose(gl);
    // floor.dispose(gl);
    textures.destroy(gl);
    // robot.dispose(gl);
  }
  /* I declare that this code is my own work */
/* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */
  //**************************************************** */
  /* Useful stuff*/
  private double startTime;
  private boolean robot1Animation = false;
  private boolean robot2Animation = true;


  public void robot1Animation(){
    if (robot1Animation){
      robot1Animation = false;
    } else{
      robot1Animation = true;
    }
  }
  public void robot2Animation(){
    if (robot2Animation){
      robot2Animation = false;
    } else{
      robot2Animation = true;
    }
  }

  

   



  private double getSeconds() {
    return System.currentTimeMillis()/1000.0;
  }
  
  // 
  // ***************************************************
  /* THE SCENE
   * Now define all the methods to handle the scene.
   * This will be added to in later examples.
   */
  // texture id
  private TextureLibrary textures;
  private Texture robot1Map, robot1Map2, robot2Map;
  private Room room;
  private GlobeStand globeStand;
  private Light light, spotLight;
  private Vec3 ROBOT2_INITIAL_POS =  new Vec3(-3.45f, 0.5f, 3.0f); 
  private Vec3 ROBOT1_INITIAL_POS =  new Vec3(-2.5f,0.5f,-6.0f); 
  private Vec3 GLOBE_INITIAL_POS = new Vec3(2.0f, 0.5f, 5.0f);
  private float [] ROOM_DIMs = {16f,16f};
  private DancingRobot robot1;
  private SurveyRobot robot2;


  public void lightSlider(float brightness) {
    light.updateBrightness(brightness);
  }
  
  
  public void initialise(GL3 gl) {
    createRandomNumbers();
    textures = new TextureLibrary();
    textures.add(gl, "chequerboard", "assets/textures/chequerboard.jpg");
    textures.add(gl, "container", "assets/textures/container2.jpg");
    textures.add(gl, "specular_container", "assets/textures/container2_specular.jpg");
    textures.add(gl, "jade", "assets/textures/jade_floor.jpg");
    textures.add(gl, "jade_specular", "assets/textures/jade_specular.jpg");
    textures.add(gl, "globe", "assets/textures/globe.jpg");
    textures.add(gl, "stars", "assets/textures/stars.jpg");
    textures.add(gl, "diffuse_file", "assets/textures/diffuse_file.jpg");
    textures.add(gl, "marble2", "assets/textures/jup0vss1.jpg");
    textures.add(gl, "specular_file", "assets/textures/specular_file.jpg");
    textures.add(gl, "cloud", "assets/textures/cloud.jpg");
    textures.add(gl, "axis", "assets/textures/axis.jpg");
    textures.add(gl, "marble", "assets/textures/white_marble_diffuse.jpg");
    textures.add(gl, "black", "assets/textures/black.jpg");
    textures.add(gl, "metal", "assets/textures/metal.jpg");
    robot1Map = textures.get("marble");
    robot1Map2 = textures.get("black");
    robot2Map = textures.get("metal");
    light = new Light(gl);
    spotLight = new Light(gl);
    light.setCamera(camera);
    room = new Room(gl, ROOM_DIMs, camera, light, textures);
    globeStand = new GlobeStand(gl, GLOBE_INITIAL_POS, camera, light, textures);
    robot1 = new DancingRobot(gl, camera, light, robot1Map, robot1Map2 , ROBOT1_INITIAL_POS);
    robot2 = new SurveyRobot(gl, camera, spotLight, robot2Map, robot2Map,ROBOT2_INITIAL_POS, ROBOT1_INITIAL_POS);
  }
  private void render(GL3 gl) {
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    light.setPosition(new Vec3(0f, 8f, 0f)); // Fixed position
    room.render(gl);
    globeStand.render(gl);
    light.render(gl);
    
    if (robot2Animation) {
      robot2.updateRobotPosition();
      if (robot2.isNear){
        robot1.updateAnimation();
      }
    } 
    if (robot1Animation){
      robot1.updateAnimation();
    }
    robot1.render(gl);
    robot2.render(gl);
    System.out.println("");
  }
  // ***************************************************
  /* An array of random numbers
   */ 
  
  private int NUM_RANDOMS = 1000;
  private float[] randoms;
  
  private void createRandomNumbers() {
    randoms = new float[NUM_RANDOMS];
    for (int i=0; i<NUM_RANDOMS; ++i) {
      randoms[i] = (float)Math.random();
    }
  }

}