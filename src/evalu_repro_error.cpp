#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

#include <fstream>
#include <glob.h>
#include <string>

using namespace cv;
using namespace std;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

// Calculate the reprojection error based on detected keypoints
static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

// Detect different type of corners
static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch(patternType)
    {
      case CHESSBOARD:
      case CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                          float(i*squareSize), 0));
        break;

      case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

      default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

// Input camera information, for example: image height, width and so on..
static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       const vector<Point3f>& newObjPoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if( !newObjPoints.empty() )
    {
        fs << "grid_points" << newObjPoints;
    }
}

int main( int argc, char** argv )
{
    // Read camera parameters
    string cam_dir_path;
    cam_dir_path.append(argv[1]);
    cam_dir_path.append("camera.txt");
    ifstream inFile(cam_dir_path.c_str());
    
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        // call system to stop
        exit(1);
    }

    int img_width, img_height;
    int count= 0, width_raw, height_raw, cam_matrix_raw, dis_coeff_raw;

    vector<float> cam_mat_vector;
    vector<float> dis_coeff_vector;

    string str_line;
    while (getline(inFile, str_line))
    {
        // Read image width
        if (str_line=="width")
            width_raw = count+1;
        if (count==width_raw)
            istringstream(str_line) >> img_width;
        
        // Read image height
        if (str_line=="height")
            height_raw = count+1;
        if (count==height_raw)
            istringstream(str_line) >> img_height;
        
        // Read camera matrix
        if (str_line=="camera matrix")
            cam_matrix_raw = count+1;
        if (count==cam_matrix_raw || count==cam_matrix_raw+1 || count==cam_matrix_raw+2)
        {
            float cam_value;
            stringstream cam_mat_string(str_line);
            while (cam_mat_string>>cam_value)
                cam_mat_vector.push_back(cam_value);
        }

        // Read distortion coefficients
        if (str_line=="distortion")
            dis_coeff_raw = count+1;
        if (count==dis_coeff_raw)
        {
            float dis_value;
            stringstream dis_coeff_string(str_line);
            while (dis_coeff_string>>dis_value)
                dis_coeff_vector.push_back(dis_value);
        }
        count++;
    }

    // Input camera intrinsics
    Size imageSize(img_width, img_height);
    float squareSize = stof(argv[2]);
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, cam_mat_vector.data());
    Mat distCoeffs   = Mat(1, 8, CV_32FC1, dis_coeff_vector.data());
    string outputFilename;
    string inputFilename = "";

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;

    bool flipVertical;
    bool showUndistorted;
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;

    int corn_size_row = stoi(argv[3]);
    int corn_size_col = stoi(argv[4]);
    Size patternsize(corn_size_row, corn_size_col);

    // TODO: read all image or manual choose one?
/*  // Read image
    vector<String> fn;
    string img_dir_path;
    img_dir_path.append(argv[1]);
    img_dir_path.append)("*.png");
    glob(img_dir_path, fn, false);

    vector<Mat> images;
    //number of png files in images folder
    size_t count = fn.size();
    for (size_t i=0; i<count; i++)
    {
        images.push_back(imread(fn[i]));
    }
*/

    // Manual choose a calibrated image
    string img_dir_path;
    img_dir_path.append(argv[1]);
    img_dir_path.append(argv[5]);
    Mat img = imread(img_dir_path, CV_LOAD_IMAGE_GRAYSCALE);

    vector<Point2f> corners;

    bool pattern_found = findChessboardCorners(img, patternsize, corners, 
                                               CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    
    if(pattern_found) {
        cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.0001));
    } 
    //drawChessboardCorners(img, patternsize, corners, pattern_found);

    // Read corners in checkboard
    vector<Point3f> boardpoints(corn_size_row * corn_size_col);
    for (int col=0; col<corn_size_col; col++)
    {
        for (int row=0; row<corn_size_row; row++)
        {
            boardpoints[col*corn_size_row + row].x = row * squareSize;
            boardpoints[col*corn_size_row + row].y = col * squareSize;
            boardpoints[col*corn_size_row + row].z = 0.0000;
        }
    }

    // Calculate the camera's pose through PnP
    Mat rvec, tvec;
    bool extrin_mat_found = solvePnP(boardpoints, corners, cameraMatrix, distCoeffs, rvec, tvec, false);
    
    // Reproject the corners on chessboard to image
    vector<Point2f> imagePoints2;
    double err;
    projectPoints(boardpoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints2);

    // Remove the disdortion of the image
    Mat tmp;
    undistort(img, tmp, cameraMatrix, distCoeffs);
    
    // Detect corners in image
    vector<Point2f> new_corners;
    bool new_pattern_found = findChessboardCorners(tmp, patternsize, new_corners, 
                                                   CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    
    // Calculate the corners' position in image
    if(new_pattern_found) {
        cornerSubPix(tmp, new_corners, Size(11, 11), Size(-1, -1),
                     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.0001));
    }
    //drawChessboardCorners(img, patternsize, new_corners, new_pattern_found);

    // Display both the reprojected points && corners in image
    Mat color_img;
    cvtColor(img, color_img, COLOR_GRAY2RGB);
    for (int i=0; i<(corn_size_row*corn_size_col); i++)
    {
        circle(color_img, imagePoints2[i], 4, Scalar(0, 0, 255), 8, CV_AA);
	    circle(color_img, new_corners[i], 4, Scalar(0, 255, 0), 8, CV_AA);
    }

    // Calculate the reprojection errors
    err = norm(new_corners, imagePoints2, NORM_L2);
    cout<< "calibration error of this image: " << err << endl;

    // Show the result
    namedWindow("Display result", WINDOW_AUTOSIZE);
    imshow("Display result", color_img);
    waitKey(0);

    return 0;
}

