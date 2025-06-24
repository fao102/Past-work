

import gmaths.*;

import java.nio.*;
import com.jogamp.common.nio.*;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*;
import com.jogamp.opengl.util.awt.*;
import com.jogamp.opengl.util.glsl.*;
import com.jogamp.opengl.util.texture.*;
import com.jogamp.opengl.util.texture.awt.*;
import com.jogamp.opengl.util.texture.spi.JPEGImage;
import com.jogamp.opengl.*;

public class Spotlight {
    private SGNode node;  // Node in the scene graph
    private float cutoffAngle; // Spotlight cone angle

//     public Spotlight(SGNode node, float cutoffAngle) {
//         this.node = node;
//         this.cutoffAngle = cutoffAngle;
//     }

//     public Vec3 getPosition() {
//         Mat4 worldTransform = node.worldTransform;
//         return new Vec3(worldTransform.[3][0], worldTransform.m[3][1], worldTransform.m[3][2]);
//     }

//     public Vec3 getDirection() {
//         Mat4 worldTransform = node.worldTransform;
//         Vec4 defaultDirection = new Vec3(0f, 0f, -1f); // Spotlight points down -Z by default
//         Vec4 transformedDirection = Mat4.multiply(worldTransform, defaultDirection);
//         return new Vec3(transformedDirection.x, transformedDirection.y, transformedDirection.z).normalize();
//     }

//     public float getCutoffAngle() {
//         return cutoffAngle;
//     }
}
