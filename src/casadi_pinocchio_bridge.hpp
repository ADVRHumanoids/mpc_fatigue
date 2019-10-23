#include <eigen3/Eigen/Dense>
#include <casadi/casadi.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>

#include <urdf_parser/urdf_parser.h>

typedef casadi::SX Scalar; // define SX as scalar type for pinocchio
typedef Eigen::Matrix<Scalar, -1, 1>  VectorXs; // a vector of SX
typedef Eigen::Matrix<Scalar, -1, -1> MatrixXs; // a matrix of SX

/**
 * Convenience functions to convert between eigen3 and casadi
 * vector/matrix representations
 */
VectorXs cas_to_eig(const casadi::SX& cas)
{
    VectorXs eig(cas.size1());
    for(int i = 0; i < eig.size(); i++)
    {
        eig(i) = cas(i);
    }
    return eig;
}

casadi::SX eig_to_cas(const VectorXs& eig)
{
    auto sx = casadi::SX(casadi::Sparsity::dense(eig.size()));
    for(int i = 0; i < eig.size(); i++)
    {
        sx(i) = eig(i);
    }
    return sx;

}

casadi::SX eigmat_to_cas(const MatrixXs& eig)
{
    auto sx = casadi::SX(casadi::Sparsity::dense(eig.rows(), eig.cols()));
    for(int i = 0; i < eig.rows(); i++)
    {
        for(int j = 0; j < eig.cols(); j++)
        {
            sx(i,j) = eig(i,j);
        }
    }
    return sx;

}



std::string generate_inv_dyn(std::string urdf_string)
{

    auto urdf = urdf::parseURDF(urdf_string);

    pinocchio::Model model_dbl;
    pinocchio::urdf::buildModel(urdf, model_dbl, true);
    pinocchio::Data data_dbl(model_dbl);

    auto model = model_dbl.cast<Scalar>();
    pinocchio::DataTpl<Scalar> data(model);
    int nq = model.nq;
    int nv = model.nv;

    // casadi variabes
    casadi::SX u = casadi::SX::sym("u", nv); // control (qddot)
    casadi::SX q = casadi::SX::sym("q", nq), qdot = casadi::SX::sym("qdot", nv); // states

    // Compute expression for inverse dynamics with Pinocchio
    pinocchio::rnea(model, data, cas_to_eig(q), cas_to_eig(qdot), cas_to_eig(u));
    auto tau = eig_to_cas(data.tau);
    casadi::Function ID("inverse_dynamics", {q, qdot, u}, {tau}, {"q", "qdot", "qddot"}, {"tau"});

    std::stringstream ss;
    ss << ID.serialize();

    return ss.str();

}

std::string generate_forward_kin(std::string urdf_string, std::string body_name)
{

    auto urdf = urdf::parseURDF(urdf_string);

    pinocchio::Model model_dbl;
    pinocchio::urdf::buildModel(urdf, model_dbl, true);
    pinocchio::Data data_dbl(model_dbl);

    auto model = model_dbl.cast<Scalar>();
    pinocchio::DataTpl<Scalar> data(model);
    int nq = model.nq;

    // casadi variabes
    casadi::SX q = casadi::SX::sym("q", nq);

    auto frame_idx = model.getFrameId(body_name);

    // Compute expression for forward kinematics with Pinocchio
    pinocchio::framesForwardKinematics(model, data, cas_to_eig(q));
    auto eig_fk_pos = data.oMf.at(frame_idx).translation();
    auto eig_fk_rot = data.oMf.at(frame_idx).rotation();
    auto ee_position = eig_to_cas(eig_fk_pos);
    auto ee_rot = eigmat_to_cas(eig_fk_rot);
    casadi::Function FK("forward_kinematics", {q}, {ee_position, ee_rot}, {"q"}, {"ee_pos", "ee_rot"});

    std::stringstream ss;
    ss << FK.serialize();

    return ss.str();
}

std::string generate_jacobian(std::string urdf_string, std::string body_name)
{

    auto urdf = urdf::parseURDF(urdf_string);

    pinocchio::Model model_dbl;
    pinocchio::urdf::buildModel(urdf, model_dbl, true);
    pinocchio::Data data_dbl(model_dbl);

    auto model = model_dbl.cast<Scalar>();
    pinocchio::DataTpl<Scalar> data(model);
    int nq = model.nq;

    // casadi variabes
    casadi::SX q = casadi::SX::sym("q", nq);

    auto frame_idx = model.getFrameId(body_name);

    // Compute expression for forward kinematics with Pinocchio
    Eigen::Matrix<Scalar, 6, -1> J;
    int nv = model.nv;
    J.setZero(6, nv);
    pinocchio::computeJointJacobians(model, data, cas_to_eig(q));
    pinocchio::framesForwardKinematics(model, data, cas_to_eig(q));
    //pinocchio::frameJacobian(model, data, cas_to_eig(q),frame_idx, J);
    pinocchio::getFrameJacobian(model, data,frame_idx, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
    auto Jac = eigmat_to_cas(J);
    casadi::Function JACOBIAN("jacobian", {q}, {Jac}, {"q"}, {"J"});

    std::stringstream ss;
    ss << JACOBIAN.serialize();

    return ss.str();

}
