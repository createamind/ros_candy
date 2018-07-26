// Generated by gencpp from file candy/UpdateWeightsResponse.msg
// DO NOT EDIT!


#ifndef CANDY_MESSAGE_UPDATEWEIGHTSRESPONSE_H
#define CANDY_MESSAGE_UPDATEWEIGHTSRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace candy
{
template <class ContainerAllocator>
struct UpdateWeightsResponse_
{
  typedef UpdateWeightsResponse_<ContainerAllocator> Type;

  UpdateWeightsResponse_()
    : b()  {
    }
  UpdateWeightsResponse_(const ContainerAllocator& _alloc)
    : b(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _b_type;
  _b_type b;





  typedef boost::shared_ptr< ::candy::UpdateWeightsResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::candy::UpdateWeightsResponse_<ContainerAllocator> const> ConstPtr;

}; // struct UpdateWeightsResponse_

typedef ::candy::UpdateWeightsResponse_<std::allocator<void> > UpdateWeightsResponse;

typedef boost::shared_ptr< ::candy::UpdateWeightsResponse > UpdateWeightsResponsePtr;
typedef boost::shared_ptr< ::candy::UpdateWeightsResponse const> UpdateWeightsResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::candy::UpdateWeightsResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::candy::UpdateWeightsResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace candy

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::candy::UpdateWeightsResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::candy::UpdateWeightsResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::candy::UpdateWeightsResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "7ce4159d4691541e9099927d38b0b65f";
  }

  static const char* value(const ::candy::UpdateWeightsResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x7ce4159d4691541eULL;
  static const uint64_t static_value2 = 0x9099927d38b0b65fULL;
};

template<class ContainerAllocator>
struct DataType< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "candy/UpdateWeightsResponse";
  }

  static const char* value(const ::candy::UpdateWeightsResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string b\n\
";
  }

  static const char* value(const ::candy::UpdateWeightsResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.b);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct UpdateWeightsResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::candy::UpdateWeightsResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::candy::UpdateWeightsResponse_<ContainerAllocator>& v)
  {
    s << indent << "b: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.b);
  }
};

} // namespace message_operations
} // namespace ros

#endif // CANDY_MESSAGE_UPDATEWEIGHTSRESPONSE_H
