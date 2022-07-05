import mujoco_py
import os


def main():
	xml_path = os.path.join(os.path.dirname(__file__), 'demo_scissor.xml')
	model = mujoco_py.load_model_from_path(xml_path)
	sim = mujoco_py.MjSim(model)
	# viewer = mujoco_py.MjViewerBasic(sim)
	viewer = mujoco_py.MjViewer(sim)

	i = 1
	while True:
		viewer.render()

		if i % (144/2) == 0:
			sim.step()

		i += 1


if __name__ == "__main__":
	main()
	