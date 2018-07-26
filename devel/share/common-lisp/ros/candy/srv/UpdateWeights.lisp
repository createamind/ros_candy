; Auto-generated. Do not edit!


(cl:in-package candy-srv)


;//! \htmlinclude UpdateWeights-request.msg.html

(cl:defclass <UpdateWeights-request> (roslisp-msg-protocol:ros-message)
  ((a
    :reader a
    :initarg :a
    :type cl:string
    :initform ""))
)

(cl:defclass UpdateWeights-request (<UpdateWeights-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UpdateWeights-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UpdateWeights-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<UpdateWeights-request> is deprecated: use candy-srv:UpdateWeights-request instead.")))

(cl:ensure-generic-function 'a-val :lambda-list '(m))
(cl:defmethod a-val ((m <UpdateWeights-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:a-val is deprecated.  Use candy-srv:a instead.")
  (a m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UpdateWeights-request>) ostream)
  "Serializes a message object of type '<UpdateWeights-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'a))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'a))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UpdateWeights-request>) istream)
  "Deserializes a message object of type '<UpdateWeights-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'a) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'a) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UpdateWeights-request>)))
  "Returns string type for a service object of type '<UpdateWeights-request>"
  "candy/UpdateWeightsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateWeights-request)))
  "Returns string type for a service object of type 'UpdateWeights-request"
  "candy/UpdateWeightsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UpdateWeights-request>)))
  "Returns md5sum for a message object of type '<UpdateWeights-request>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UpdateWeights-request)))
  "Returns md5sum for a message object of type 'UpdateWeights-request"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UpdateWeights-request>)))
  "Returns full string definition for message of type '<UpdateWeights-request>"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UpdateWeights-request)))
  "Returns full string definition for message of type 'UpdateWeights-request"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UpdateWeights-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'a))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UpdateWeights-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UpdateWeights-request
    (cl:cons ':a (a msg))
))
;//! \htmlinclude UpdateWeights-response.msg.html

(cl:defclass <UpdateWeights-response> (roslisp-msg-protocol:ros-message)
  ((b
    :reader b
    :initarg :b
    :type cl:string
    :initform ""))
)

(cl:defclass UpdateWeights-response (<UpdateWeights-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UpdateWeights-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UpdateWeights-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<UpdateWeights-response> is deprecated: use candy-srv:UpdateWeights-response instead.")))

(cl:ensure-generic-function 'b-val :lambda-list '(m))
(cl:defmethod b-val ((m <UpdateWeights-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:b-val is deprecated.  Use candy-srv:b instead.")
  (b m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UpdateWeights-response>) ostream)
  "Serializes a message object of type '<UpdateWeights-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'b))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'b))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UpdateWeights-response>) istream)
  "Deserializes a message object of type '<UpdateWeights-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'b) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'b) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UpdateWeights-response>)))
  "Returns string type for a service object of type '<UpdateWeights-response>"
  "candy/UpdateWeightsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateWeights-response)))
  "Returns string type for a service object of type 'UpdateWeights-response"
  "candy/UpdateWeightsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UpdateWeights-response>)))
  "Returns md5sum for a message object of type '<UpdateWeights-response>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UpdateWeights-response)))
  "Returns md5sum for a message object of type 'UpdateWeights-response"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UpdateWeights-response>)))
  "Returns full string definition for message of type '<UpdateWeights-response>"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UpdateWeights-response)))
  "Returns full string definition for message of type 'UpdateWeights-response"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UpdateWeights-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'b))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UpdateWeights-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UpdateWeights-response
    (cl:cons ':b (b msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UpdateWeights)))
  'UpdateWeights-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UpdateWeights)))
  'UpdateWeights-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateWeights)))
  "Returns string type for a service object of type '<UpdateWeights>"
  "candy/UpdateWeights")