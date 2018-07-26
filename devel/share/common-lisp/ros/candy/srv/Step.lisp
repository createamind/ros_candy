; Auto-generated. Do not edit!


(cl:in-package candy-srv)


;//! \htmlinclude Step-request.msg.html

(cl:defclass <Step-request> (roslisp-msg-protocol:ros-message)
  ((a
    :reader a
    :initarg :a
    :type cl:string
    :initform ""))
)

(cl:defclass Step-request (<Step-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Step-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Step-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<Step-request> is deprecated: use candy-srv:Step-request instead.")))

(cl:ensure-generic-function 'a-val :lambda-list '(m))
(cl:defmethod a-val ((m <Step-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:a-val is deprecated.  Use candy-srv:a instead.")
  (a m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Step-request>) ostream)
  "Serializes a message object of type '<Step-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'a))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'a))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Step-request>) istream)
  "Deserializes a message object of type '<Step-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Step-request>)))
  "Returns string type for a service object of type '<Step-request>"
  "candy/StepRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step-request)))
  "Returns string type for a service object of type 'Step-request"
  "candy/StepRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Step-request>)))
  "Returns md5sum for a message object of type '<Step-request>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Step-request)))
  "Returns md5sum for a message object of type 'Step-request"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Step-request>)))
  "Returns full string definition for message of type '<Step-request>"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Step-request)))
  "Returns full string definition for message of type 'Step-request"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Step-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'a))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Step-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Step-request
    (cl:cons ':a (a msg))
))
;//! \htmlinclude Step-response.msg.html

(cl:defclass <Step-response> (roslisp-msg-protocol:ros-message)
  ((b
    :reader b
    :initarg :b
    :type cl:string
    :initform ""))
)

(cl:defclass Step-response (<Step-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Step-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Step-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<Step-response> is deprecated: use candy-srv:Step-response instead.")))

(cl:ensure-generic-function 'b-val :lambda-list '(m))
(cl:defmethod b-val ((m <Step-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:b-val is deprecated.  Use candy-srv:b instead.")
  (b m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Step-response>) ostream)
  "Serializes a message object of type '<Step-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'b))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'b))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Step-response>) istream)
  "Deserializes a message object of type '<Step-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Step-response>)))
  "Returns string type for a service object of type '<Step-response>"
  "candy/StepResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step-response)))
  "Returns string type for a service object of type 'Step-response"
  "candy/StepResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Step-response>)))
  "Returns md5sum for a message object of type '<Step-response>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Step-response)))
  "Returns md5sum for a message object of type 'Step-response"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Step-response>)))
  "Returns full string definition for message of type '<Step-response>"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Step-response)))
  "Returns full string definition for message of type 'Step-response"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Step-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'b))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Step-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Step-response
    (cl:cons ':b (b msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Step)))
  'Step-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Step)))
  'Step-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step)))
  "Returns string type for a service object of type '<Step>"
  "candy/Step")