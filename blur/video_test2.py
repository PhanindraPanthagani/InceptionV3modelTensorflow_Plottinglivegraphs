from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import cv2
from imutils.video import FileVideoStream

from imutils import video
import time
import os
from colorama import Fore,init


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['axes.grid'] = True
#plt.ion()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')
parser.add_argument("-v", "--video", required=True,
                help="path to input video file")
args = vars(parser.parse_args())


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(labels, input_layer_name, output_layer_name,num_top_predictions):
    f1 = open('./result_file.txt', 'w+')

    with tf.Session() as sess:
        #video_capture = cv2.VideoCapture(args["video"])
        #video_capture = cv2.VideoCapture(0)

        fps_video =video.FPS().start()
        print("[INFO] starting video file thread...")
        fvs = FileVideoStream(args["video"]).start()
        time.sleep(1.0)

        i = 0


        Blur=[]
        Non_Blur=[]

        dirname = 'video_frames'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        #os.mkdir(dirname)

        predicted_names=[]
        predicted_values=[]


        while fvs.more(): # fps._numFrames < 120
                frame = fvs.read()  # get current frame
            #if (b == True):  # not necessary
                i = i + 1
                filename = "frame" + str(i) + ".png"
                cv2.imwrite(os.path.join(dirname,filename), img=frame)

                #cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #cv2.imwrite(filename="screens" + str(i) + "alpha.png", img=cv2.resize(frame, (640,320), interpolation = cv2.INTER_AREA));  # write frame image to file

                image_data = tf.gfile.FastGFile((os.path.join(dirname,filename)), 'rb').read()  # get this image file
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor, \
                                     {'DecodeJpeg/contents:0': image_data})  # analyse the image
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]


                predicted_names.append(labels[top_k[0]])
                predicted_values.append(predictions[0][top_k[0]])

                print (i, file=f1)
                print(labels[top_k[0]], file=f1)
                print(predictions[0][top_k[0]], file=f1)
                print("\n", file=f1)

                # f1.write(labels[top_k[0]])
                # f1.write(predictions[0][top_k[0]])

                for node_id in top_k:

                    name_string = labels[node_id]
                    score = predictions[0][node_id]


                    # init()
                    # #init(convert=True)
                    # if (name_string=="blur" and score >= 0.70):
                    #     print(Fore.RED+'%s (score = %.5f)' % (name_string, score))
                    # elif (name_string=="non blur" and score >= 0.70):
                    #     print(Fore.GREEN+'%s (score = %.5f)' % (name_string, score))
                    # else:
                    #     print(Fore.BLACK+'%s (score = %.5f)' % (name_string, score))






                    print( '%s (score = %.5f)' % (name_string, score))
                    if (node_id==0):
                          Blur.append(score)
                          plt.figure(1)
                          plt.title("Blur")
                          plt.xlabel("Frame Number")
                          plt.ylabel("Prediction score")
                          plt.scatter(i, score,color=['red'])
                          plt.pause(0.001)




                    elif (node_id == 1):
                          Non_Blur.append(score)
                          plt.figure(2)
                          plt.title("Non Blur")
                          plt.xlabel("Frame Number")
                          plt.ylabel("Prediction score")
                          plt.scatter(i, score,color=['green'])
                          plt.pause(0.001)





                print("\n\n Frame number ",i)
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 600,600)
                cv2.imshow("image", frame)  # show frame in window
                cv2.waitKey(1)  # wait 1ms -> 0 until key input
                fps_video.update()


        print ("Total frames" , i)



        #plt.figure(2)
        f, axarr = plt.subplots(2, sharex=True,sharey=True)
        t1 = np.arange(0.0, i, 1.0)
        tick_spacing = 0.25
        plt.xlabel('FRAME NUMBERS',fontsize=14)
        plt.ylabel('               PREDICTED SCORES (0 to 1)',fontsize=14)


        axarr[0].plot(t1, Blur)
        axarr[0].set_title('Blur_screen')
        axarr[0].set_ylim([0,1.0])
        axarr[0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        axarr[1].plot(t1,Non_Blur)
        axarr[1].set_title('Non blur screen')
        axarr[1].set_ylim([0,1.0])
        axarr[1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



        fps_video.stop()
        print("[INFO] elasped time: {:.2f}".format(fps_video.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps_video.fps()))


        #input()

        plt.show()
        cv2.destroyAllWindows()
        f1.close()

        return 0


def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])


  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)


  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  run_graph(labels, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
