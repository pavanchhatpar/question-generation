import argparse
import imageio
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, dest='input')
    parser.add_argument("-o", "--output", type=str, required=True, dest='output')
    args = parser.parse_args()
    return args.input, args.output


def create_gif(input, output):
    anim_file = output
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('{}/generated_plot_*.png'.format(input))
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    gif = imageio.mimread(output)
    imageio.mimsave(output, gif, fps=1)

if __name__ == "__main__":
    input, output = parse_args()
    create_gif(input, output)