import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'api.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Obtain a list of the available cameras on the device.
  final cameras = await availableCameras();
  // Get a specific camera from the list of available cameras.
  final camera = cameras.first;
  runApp(
    MaterialApp(
        debugShowCheckedModeBanner: false,
        title: "Camera App",
        home: HomePage(camera: camera,)
    )
  );
}

class HomePage extends StatefulWidget {
  final CameraDescription camera;
  const HomePage({
    Key key,
    @required this.camera,
  }) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List _imgBytes;
  CameraController _cameraController;
  Future<void> _initializeCameraControllerFuture;
  CloudApi api;

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController(widget.camera, ResolutionPreset.high);
    _initializeCameraControllerFuture = _cameraController.initialize();
    rootBundle.loadString('assets/credentials.json').then((json) {
      api = CloudApi(json);
    });
  }

  void _showCamera() async {
    try {
      await _initializeCameraControllerFuture;
      final picture = await _cameraController.takePicture();
      _imgBytes = await picture.readAsBytes();
      String base64encode = base64.encode(_imgBytes);
      Map<String, dynamic> d = {'name': picture.name, 'img': base64encode};
      String result = json.encode(d);
      await http.post("http://10.0.2.2:5000/", body: result);
      //_saveImage(picture.name);
    } catch (e) {
      print(e);
    }
  }

  void _saveImage(String name) async{
    final response = await api.save(name, _imgBytes);
    print(response.downloadLink);
  }


  @override
  Widget build(BuildContext context) {
    var screenSize = MediaQuery.of(context).size;
    return Scaffold(
        body: SafeArea(
          child: Column(children: <Widget>[
            SizedBox(height: 10,),
            FlatButton(
              height: ((screenSize.height))-50,
              minWidth: screenSize.width,
              child: Text("Take Picture", style: TextStyle(color: Colors.white, fontSize: 75), textAlign: TextAlign.center,),
              color: Colors.green,
              onPressed: () async{
                _showCamera();
              },
            ),
            SizedBox(height: 10,),
          ]),
        )
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }
}