import 'dart:typed_data';
import 'package:googleapis_auth/auth_io.dart' as auth;
import 'package:gcloud/storage.dart';
import 'package:mime/mime.dart';
import 'dart:async';

class CloudApi{
  final auth.ServiceAccountCredentials _credentials;
  auth.AutoRefreshingAuthClient _client;

  CloudApi(String json)
      : _credentials = auth.ServiceAccountCredentials.fromJson(json);

  Future<ObjectInfo> save(String name, Uint8List imgBytes) async{
    if (_client == null){
      _client = await auth.clientViaServiceAccount(_credentials, Storage.SCOPES);
    }

    var storage = Storage(_client, 'Science Fair');
    var bucket = storage.bucket('sci-fair-bucket');

    final type = lookupMimeType(name);
    return await bucket.writeBytes(name, imgBytes, metadata: ObjectMetadata(contentType: type));
  }
}