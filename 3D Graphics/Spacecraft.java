
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.FPSAnimator;


public class Spacecraft extends JFrame implements ActionListener {   // A subclass of JFrame.

  private static final int WIDTH = 1024;
  private static final int HEIGHT = 768;
  private static final Dimension dimension = new Dimension(WIDTH, HEIGHT);
                                    // dimension is declared as static so it can be used in the main method,
                                    // which is static.
  private GLCanvas canvas;          // The canvas that will be drawn on.
  private GLEventListener glEventListener;
                                    // The listener to handle GL events
  private FPSAnimator animator; 
  private Camera camera;
  public float brightness;
                                    
                                    // animator is declared as an attribute of the class so it can be accessed
                                    // and stopped from within the window closing event handler.
  
  public static void main(String[] args) {
    Spacecraft f = new Spacecraft("Spacecraft");         // Create a subclass of JFrame.
    f.getContentPane().setPreferredSize(dimension);
                                    // setPreferredSize() is used for the content pane. When setPreferredSize is
                                    // used, must remember to pack() the JFrame after all elements have been added.
                                    // Note that the JFrame will be bigger, as the borders and title bar are added.
    f.pack();                       // Without pack(), the use of setPreferredSize() would 
                                    // result in a 0x0 canvas size, as nothing is yet drawn on it.
                                    // Alternative is to use f.setSize(dimension); rather than setPreferredSize and pack.
    f.setVisible(true);             // Finally, set the JFrame to be visible.
    f.canvas.requestFocusInWindow();
  }

  public Spacecraft(String textForTitleBar) {
    super(textForTitleBar);
    GLCapabilities glcapabilities = new GLCapabilities(GLProfile.get(GLProfile.GL3));
    canvas = new GLCanvas(glcapabilities);
    camera = new Camera(Camera.DEFAULT_POSITION, Camera.DEFAULT_TARGET, Camera.DEFAULT_UP);
    glEventListener = new AppEventListener(camera);
    canvas.addGLEventListener(glEventListener);
    canvas.addMouseMotionListener(new MyMouseInput(camera));
    canvas.addKeyListener(new MyKeyboardInput(camera));
    getContentPane().add(canvas, BorderLayout.CENTER);
    
    
    JMenuBar menuBar=new JMenuBar();
    this.setJMenuBar(menuBar);
      JMenu fileMenu = new JMenu("File");
        JMenuItem quitItem = new JMenuItem("Quit");
        quitItem.addActionListener(this);
        fileMenu.add(quitItem);
    menuBar.add(fileMenu);

    /* I declare that this code is my own work */
    /* Author Filé Ajanaku fajanaku-olaleye1@sheffield.ac.uk */

    
    
    JPanel p = new JPanel();
      JLabel label1 = new JLabel();
      label1.setText("Lighting");
      p.add(label1);
      JSlider slider =new JSlider(JSlider.HORIZONTAL, 0, 100, 100); 
      slider.setMajorTickSpacing(10);
      slider.setMinorTickSpacing(1);
      slider.addChangeListener(e -> {
        int brightnessValue = slider.getValue();
        brightness = brightnessValue / 100.0f; // Convert to a range of 0.0f to 1.0f
        ((AppEventListener) glEventListener).lightSlider(brightness);
       // Update light brightness
      });
      p.add(slider);
      JButton b = new JButton("Robot 1 ON/OFF");
      b.addActionListener(this);
      p.add(b);
      b = new JButton("Robot 2 ON/OFF");
      b.addActionListener(this);
      p.add(b);
    this.add(p, BorderLayout.SOUTH);
    
    addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent e) {
        animator.stop();
        remove(canvas);
        dispose();
        System.exit(0);
      }
    });
    animator = new FPSAnimator(canvas, 60);
    animator.start();
    
  }
  
  public void actionPerformed(ActionEvent e) {
    if (e.getActionCommand().equalsIgnoreCase("Robot 1 ON/OFF")) {
      ((AppEventListener) glEventListener).robot1Animation();
      
    }
    else if(e.getActionCommand().equalsIgnoreCase("Robot 2 ON/OFF")){
      ((AppEventListener) glEventListener).robot2Animation(); 
    }
    
      
  }
  
  
}

class MyKeyboardInput extends KeyAdapter  {
  private Camera camera;
  
  public MyKeyboardInput(Camera camera) {
    this.camera = camera;
  }
  
  public void keyPressed(KeyEvent e) {
    Camera.Movement m = Camera.Movement.NO_MOVEMENT;
    switch (e.getKeyCode()) {
      case KeyEvent.VK_LEFT:  m = Camera.Movement.LEFT;  break;
      case KeyEvent.VK_RIGHT: m = Camera.Movement.RIGHT; break; 
      case KeyEvent.VK_UP:    m = Camera.Movement.UP;    break;
      case KeyEvent.VK_DOWN:  m = Camera.Movement.DOWN;  break;
      case KeyEvent.VK_A:  m = Camera.Movement.FORWARD;  break;
      case KeyEvent.VK_Z:  m = Camera.Movement.BACK;  break;
    }
    camera.keyboardInput(m);
    
  }
}

class MyMouseInput extends MouseMotionAdapter {
  private Point lastpoint;
  private Camera camera;
  
  public MyMouseInput(Camera camera) {
    this.camera = camera;
  }
  
    /**
   * mouse is used to control camera position
   *
   * @param e  instance of MouseEvent
   */    
  public void mouseDragged(MouseEvent e) {
    Point ms = e.getPoint();
    float sensitivity = 0.001f;
    float dx=(float) (ms.x-lastpoint.x)*sensitivity;
    float dy=(float) (ms.y-lastpoint.y)*sensitivity;
    if (e.getModifiersEx()==MouseEvent.BUTTON1_DOWN_MASK)
      camera.updateYawPitch(dx, -dy);
    lastpoint = ms;
  }

  /**
   * mouse is used to control camera position
   *
   * @param e  instance of MouseEvent
   */  
  public void mouseMoved(MouseEvent e) {   
    lastpoint = e.getPoint(); 
  }
}