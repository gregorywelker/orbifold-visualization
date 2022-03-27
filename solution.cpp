// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Welker Gergo
// Neptun : ECKCA4
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;						
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			
	out vec4 fragmentColor;		

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

GPUProgram gpuProgram;
unsigned int vao;

class Canvas
{
	unsigned int vao;

public:
	Canvas() {}
	Canvas(int windowWidth, int windowHeight)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Display(std::vector<vec4> image)
	{
		Texture texture = Texture(windowWidth, windowHeight, image);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

Canvas canvas;

struct Ray
{
	vec3 start, dir;
	Ray(vec3 start, vec3 dir) : start(start), dir(normalize(dir)) {}
};

enum MaterialType
{
	ROUGH,
	REFLECTIVE
};

struct Material
{
	vec3 ambient, diffuse, specular, fresnel;
	float shine;
	MaterialType type;
	Material(MaterialType type) : type(type) {}
};

struct RoughMaterial : Material
{
	RoughMaterial(vec3 diffuse, vec3 specular, float shine) : Material(ROUGH)
	{
		this->ambient = diffuse * M_PI;
		this->diffuse = diffuse;
		this->specular = specular;
		this->shine = shine;
	}
};

vec3 operator/(vec3 num, vec3 denom)
{
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material
{
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE)
	{
		vec3 one(1.0f, 1.0f, 1.0f);
		fresnel = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit
{
	float t;
	vec3 position, normal;
	Material *material;
	bool portalHit;
	vec3 portalCenter;
	Hit()
	{
		t = -1;
		portalHit = false;
		portalCenter = vec3(0.0f, 0.0f, 0.0f);
	}
};

class Intersectable
{
protected:
	Material *material;

public:
	virtual Hit Intersect(const Ray &ray) = 0;
};

static bool SphereHit(const Ray &ray, vec3 center, float radius)
{
	vec3 dist = ray.start - center;
	float a = dot(ray.dir, ray.dir);
	float b = dot(dist, ray.dir) * 2.0f;
	float c = dot(dist, dist) - radius * radius;
	float discr = b * b - 4.0f * a * c;
	if (discr < 0)
		return false;
	float sqrt_discr = sqrtf(discr);
	float t1 = (-b + sqrt_discr) / 2.0f / a;
	if (t1 <= 0)
	{
		return false;
	}
	return true;
}

class Dodecaeder : public Intersectable
{
	std::vector<vec3> vertices = {
		vec3(0.0f, 0.618f, 1.618f),
		vec3(0.0f, -0.618f, 1.618f),
		vec3(0.0f, -0.618f, -1.618f),
		vec3(0.0f, 0.618f, -1.618f),
		vec3(1.618f, 0.0f, 0.618f),
		vec3(-1.618f, 0.0f, 0.618f),
		vec3(-1.618f, 0.0f, -0.618f),
		vec3(1.618f, 0.0f, -0.618f),
		vec3(0.618f, 1.618f, 0.0f),
		vec3(-0.618f, 1.618f, 0.0f),
		vec3(-0.618f, -1.618f, 0.0f),
		vec3(0.618f, -1.618f, 0.0f),
		vec3(1.0f, 1.0f, 1.0f),
		vec3(-1.0f, 1.0f, 1.0f),
		vec3(-1.0f, -1.0f, 1.0f),
		vec3(1.0f, -1.0f, 1.0f),
		vec3(1.0f, -1.0f, -1.0f),
		vec3(1.0f, 1.0f, -1.0f),
		vec3(-1.0f, 1.0f, -1.0f),
		vec3(-1.0f, -1.0f, -1.0f),
	};
	std::vector<std::vector<int>> faces = {
		{1, 2, 16, 5, 13},
		{1, 13, 9, 10, 14},
		{1, 14, 6, 15, 2},
		{2, 15, 11, 12, 16},
		{3, 4, 18, 8, 17},
		{3, 17, 12, 11, 20},
		{3, 20, 7, 19, 4},
		{19, 10, 9, 18, 4},
		{16, 12, 17, 8, 5},
		{5, 8, 18, 9, 13},
		{14, 10, 19, 7, 6},
		{6, 7, 20, 11, 15},
	};

public:
	Dodecaeder(Material *material)
	{
		this->material = material;
	}
	Hit Intersect(const Ray &ray)
	{
		Hit hit;

		if (!SphereHit(ray, vec3(0.0f, 0.0f, 0.0f), sqrtf(3.0f)))
		{
			return hit;
		}

		for (size_t i = 0; i < faces.size(); i++)
		{
			vec3 p1 = vertices[faces[i][0] - 1];
			vec3 n = normalize(cross(vertices[faces[i][1] - 1] - p1, vertices[faces[i][2] - 1] - p1));
			float t = dot(p1 - ray.start, n) / dot(ray.dir, n);
			if (t > 0.0f && (hit.t < 0.0f || t < hit.t))
			{
				vec3 p = ray.start + ray.dir * t;

				std::vector<float> sideValues;

				for (size_t j = 0; j < 5; j++)
				{
					vec3 lineStart = vertices[faces[i][j] - 1];
					vec3 lineEnd = vertices[faces[i][(j + 1) % 5] - 1];
					vec3 ref = vertices[faces[i][(j + 2) % 5] - 1];

					vec3 lineDir = lineEnd - lineStart;
					vec3 pointDir = ref - lineStart;

					float correlation = dot(lineDir, pointDir);
					float alpha = acosf(correlation / (length(lineDir) * length(pointDir)));
					float refProjectionDistance = cos(alpha) * length(pointDir);
					vec3 refProjection = lineStart + normalize(lineDir) * refProjectionDistance;
					vec3 normal = normalize(ref - refProjection);

					float sideValue = normal.x * (p.x - lineStart.x) + normal.y * (p.y - lineStart.y) + normal.z * (p.z - lineStart.z);

					sideValues.push_back(sideValue);
				}

				bool inside = true;

				for (size_t j = 0; j < sideValues.size(); j++)
				{
					if (sideValues[j] < 0.0f)
					{

						inside = false;
					}
				}
				if (inside)
				{
					bool portalHit = true;

					for (size_t j = 0; j < sideValues.size(); j++)
					{
						if (sideValues[j] < 0.1f)
						{
							portalHit = false;
						}
					}
					if (portalHit)
					{
						vec3 center(0.0f, 0.0f, 0.0f);
						for (size_t j = 0; j < 5; j++)
						{
							center = center + vertices[faces[i][j] - 1];
						}
						center = center / 5.0f;
						hit.portalCenter = center;

						hit.portalHit = true;
					}
					hit.t = t;
					hit.position = p;
					hit.normal = n;
					hit.material = material;
				}
			}
		}
		return hit;
	}
};

class ImplicitSurface : public Intersectable
{
	mat4 Q;

public:
	ImplicitSurface(Material *material)
	{
		Q = mat4(
			vec4(3.5f, 0.0f, 0.0f, 0.0f),
			vec4(0.0f, 12.7f, 0.0f, 0.0f),
			vec4(0.0f, 0.0f, 0.0f, 0.6f),
			vec4(0.0f, 0.0f, 0.6f, 0.0f));
		this->material = material;
	}

	vec3 Gradient(vec3 r)
	{
		vec4 g = vec4(r.x, r.y, r.z, 1.0f) * Q * 2.0f;
		return vec3(g.x, g.y, g.z);
	}
	Hit Intersect(const Ray &ray)
	{
		Hit hit;

		if (!SphereHit(ray, vec3(0.0f, 0.0f, 0.0f), 0.3f))
		{
			return hit;
		}

		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1.0f);
		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		float a = dot(D * Q, D);
		float b = dot(S * Q, D) * 2.0f;
		float c = dot(S * Q, S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / (2.0f * a);
		float t2 = (-b - sqrt_discr) / (2.0f * a);
		if (t1 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(Gradient(hit.position));
		hit.material = material;

		return hit;
	}
};

struct PointLight
{
	vec3 position, color;
	float intensity;
	PointLight(vec3 position, vec3 color, float intensity) : position(position), color(color), intensity(intensity) {}
};

struct DirectionalLight
{
	vec3 direction;
	vec3 color;
	DirectionalLight(vec3 direction, vec3 color)
	{
		this->direction = normalize(direction);
		this->color = color;
	}
};

class Camera
{
	vec3 position, lookat, right, up;
	float fov;

public:
	void Set(vec3 position, vec3 lookat, vec3 vup, float fov)
	{
		this->position = position;
		this->lookat = lookat;
		this->fov = fov;
		vec3 w = position - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray GetRay(int X, int Y)
	{
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - position;
		return Ray(position, dir);
	}

	void Turn(float dt)
	{
		vec3 dir = position - lookat;
		position = vec3(dir.x * cos(dt) + dir.z * sin(dt), dir.y, -dir.x * sin(dt) + dir.z * cos(dt)) + lookat;
		Set(position, lookat, up, fov);
	}
};

const float epsilon = 0.0001f;

vec4 RodriguesFormula(vec4 v1, vec4 v2)
{
	vec3 d1(v1.y, v1.z, v1.w);
	vec3 d2(v2.y, v2.z, v2.w);
	vec3 prod = v1.x * d2 + v2.x * d1 + cross(d1, d2);
	return vec4(v1.x * v2.x - dot(d1, d2), prod.x, prod.y, prod.z);
}

vec3 QuaternionRotation(vec3 point, vec3 rotAxis, float angle)
{
	vec3 axis = rotAxis * sinf(angle / 2.0f);
	vec4 q(cosf(angle / 2.0f), axis.x, axis.y, axis.z);
	vec4 rotPoint(0.0f, point.x, point.y, point.z);
	vec4 qinv = vec4(cosf(angle / 2.0f), (-1.0f) * axis.x, (-1.0f) * axis.y, (-1.0f) * axis.z) / dot(q, q);

	vec4 prod1 = RodriguesFormula(q, rotPoint);
	vec4 prod2 = RodriguesFormula(prod1, qinv);

	return vec3(prod2.y, prod2.z, prod2.w);
}

class Scene
{
	std::vector<Intersectable *> intersectables;
	std::vector<DirectionalLight *> directionalLights;
	std::vector<PointLight *> pointLights;
	Camera camera;
	vec3 ambientColor;

public:
	Scene()
	{
		camera.Set(vec3(0.0, -0.8f, 1.2f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), 65.0f * M_PI / 180.0f);
		directionalLights.push_back(new DirectionalLight(vec3(2.0f, 2.5f, 0.5f), vec3(1.0f, 1.0f, 2.0f)));
		intersectables.push_back(new Dodecaeder(new RoughMaterial(vec3(0.5f, 0.25f, 0.05f), vec3(1.0f, 0.7f, 0.4f), 50.0f)));
		intersectables.push_back(new ImplicitSurface(new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f))));
		pointLights.push_back(new PointLight(vec3(0.0f, 0.6f, 0.0f), vec3(0.25f, 0.25f, 3.75f), 5.0f));
		ambientColor = vec3(0.2f, 0.35f, 0.5f);
	}

	vec3 Trace(Ray ray, int depth = 0, int portalCrosses = 0)
	{
		if (depth > 5 || portalCrosses > 5)
		{
			return ambientColor;
		}
		Hit hit = FirstIntersect(ray);
		if (hit.t < 0)
		{
			return ambientColor;
		}

		vec3 outRadiance = vec3(0.0f, 0.0f, 0.0f);
		if (hit.portalHit)
		{
			ray.dir = QuaternionRotation(ray.dir, hit.portalCenter, (72.0f * M_PI / 180.0f) * (float)(1 + portalCrosses));
			hit.position = QuaternionRotation(hit.position, hit.portalCenter, (72.0f * M_PI / 180.0f) * (float)(1 + portalCrosses));

			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			outRadiance = Trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth, portalCrosses + 1);
		}
		else
		{
			if (hit.material->type == ROUGH)
			{
				outRadiance = hit.material->ambient * ambientColor;
				for (DirectionalLight *directionalLight : directionalLights)
				{
					float cosTheta = dot(hit.normal, directionalLight->direction);
					outRadiance = outRadiance + directionalLight->color * hit.material->diffuse * cosTheta;
					vec3 halfway = normalize(-ray.dir + directionalLight->direction);
					float cosDelta = dot(hit.normal, halfway);
					outRadiance = outRadiance + directionalLight->color * hit.material->specular * powf(cosDelta, hit.material->shine);
				}
				for (PointLight *pointLight : pointLights)
				{
					vec3 dir = hit.position - pointLight->position;
					outRadiance = outRadiance + (hit.material->diffuse * pointLight->color * pointLight->intensity) / dot(dir, dir);
				}
			}
			if (hit.material->type == REFLECTIVE)
			{
				vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1.0f, 1.0f, 1.0f);
				vec3 outFresnel = hit.material->fresnel + (one - hit.material->fresnel) * pow(1.0f - cosa, 5.0f);
				outRadiance = outRadiance + Trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1, portalCrosses) * outFresnel;
			}
		}
		return outRadiance;
	}

	Hit FirstIntersect(Ray ray)
	{
		Hit bestHit;
		for (Intersectable *intersectable : intersectables)
		{
			Hit hit = intersectable->Intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
			}
		}
		if (dot(ray.dir, bestHit.normal) > 0)
		{
			bestHit.normal = bestHit.normal * (-1);
		}
		return bestHit;
	}

	void Render()
	{
		std::vector<vec4> renderImage = std::vector<vec4>(windowWidth * windowHeight);
		for (size_t y = 0; y < windowHeight; y++)
		{
#pragma omp parallel for
			for (size_t x = 0; x < windowHeight; x++)
			{
				vec3 renderColor = Trace(camera.GetRay(x, y));
				renderImage[y * windowWidth + x] = vec4(renderColor.x, renderColor.y, renderColor.z, 1.0f);
			}
		}
		canvas.Display(renderImage);
	};

	void TurnCamera(float turnAmount = 0.05f)
	{
		camera.Turn(turnAmount);
		glutPostRedisplay();
	}
};

Scene scene = Scene();

void onInitialization()
{

	glViewport(0, 0, windowWidth, windowHeight);
	canvas = Canvas(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay()
{
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY)
{
	if (key == 'd')
		glutPostRedisplay();

	scene.TurnCamera();
}

void onKeyboardUp(unsigned char key, int pX, int pY)
{
}

void onMouseMotion(int pX, int pY)
{
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY)
{
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char *buttonStat;
	switch (state)
	{
	case GLUT_DOWN:
		buttonStat = "pressed";
		break;
	case GLUT_UP:
		buttonStat = "released";
		break;
	}

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	case GLUT_MIDDLE_BUTTON:
		printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	case GLUT_RIGHT_BUTTON:
		printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	}
}

void onIdle()
{
	long time = glutGet(GLUT_ELAPSED_TIME);
}
